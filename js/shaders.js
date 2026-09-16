'use strict';
/**
 * shaders.js
 * 全WGSL(WebGPU)シェーダーソースを定数として定義
 *
 * パーティクル構造体レイアウト (48 bytes / particle):
 *   pos : vec4f  → [x, y, z, ptype]  ptype: 0=流体, 1=壁面, 2=流入
 *   vel : vec4f  → [vx, vy, vz, pressure]
 *   aux : vec4f  → [temperature, density, speed, _pad]
 */

// ─────────────────────────────────────────────────────────────────
// 共通ヘッダー (全シェーダーで使用)
// ─────────────────────────────────────────────────────────────────
const WGSL_COMMON = /* wgsl */`
// ── Params uniform (128 bytes) ────────────────────────────────────
struct Params {
  N          : u32,   // 粒子数
  HASH_SIZE  : u32,   // ハッシュテーブルサイズ
  MAX_CELL   : u32,   // セルあたり最大粒子数
  step       : u32,   // ステップカウンタ

  L0         : f32,   // 粒子間距離 [m]
  RE         : f32,   // 影響半径 [m]
  RE2        : f32,   // RE²
  N0         : f32,   // 基準粒子数密度

  LAMBDA     : f32,   // ラプラシアン係数 [m²]
  DT         : f32,   // 時間刻み [s]
  NU         : f32,   // 動粘度 [m²/s]
  ALPHA      : f32,   // 温度拡散率 [m²/s]

  G          : f32,   // 重力加速度 [m/s²]
  BETA       : f32,   // 体積膨張係数 [1/K]
  T_REF      : f32,   // 周囲温度 [°C]
  T_IN       : f32,   // 流入温度 [°C]

  RHO0       : f32,   // 空気密度 [kg/m³]
  KAPPA      : f32,   // EOS剛性係数
  CELL_SIZE  : f32,   // ハッシュセルサイズ [m] = RE
  DOM        : f32,   // 領域サイズ [m] = 2.0

  V_IN       : f32,   // 流入速度 [m/s]
  NOZZLE     : u32,   // ノズルタイプ (1〜4)
  IN_R       : f32,   // 流入半径 [m]
  IN_CX      : f32,   // 流入中心 x [m]

  IN_CY      : f32,   // 流入中心 y [m]
  HAS_CORE   : u32,   // 内部シリンダーフラグ
  CORE_R     : f32,   // 内部シリンダー半径 [m]
  T_MIN      : f32,   // 表示温度最小値

  T_MAX      : f32,   // 表示温度最大値
  V_MAX      : f32,   // 表示速度最大値
  DISP_MODE  : u32,   // 表示モード 0=温度 1=速度 2=圧力
  DOM_Z      : f32,   // z方向総高さ [m] = DOM + NOZZLE_H (2.18)
};

struct Particle {
  pos : vec4f,   // xyz=位置, w=粒子タイプ
  vel : vec4f,   // xyz=速度, w=圧力
  aux : vec4f,   // x=温度, y=数密度, z=速度大きさ, w=ノズル粒子フラグ(0/1)
  aux2: vec4f,   // x=質量比mr(体積比。室内粒子=1.0、微細化領域=(L0n/L0)³), y/z/w=予約
};

// ── 質量重み付きMPSカーネル: w(r)*mr で「体積」を揃えて多重解像度に対応 ──
// 粒子間隔が異なる領域が混在しても、体積比で重み付けすることで
// 数密度・圧力勾配・粘性項が物理的に正しいスケールに保たれる。
// (旧: 二重解像度 L0r=0.05/L0n=0.005 では質量重みなしのため
//  RE内に~700粒子が入り n_i/N0 ≈ 47 → 圧力が数百倍に発散していた)
fn mpsWm(r: f32, re: f32, mr: f32) -> f32 {
  return mpsW(r, re) * mr;
}

// ── MPS核関数 w(r) = re/r - 1 ────────────────────────────────────
fn mpsW(r: f32, re: f32) -> f32 {
  if (r < re && r > 1e-7f) {
    return re / r - 1.0f;
  }
  return 0.0f;
}

// ── 空間ハッシュ ─────────────────────────────────────────────────
fn cellHash(ci: vec3i, hashSize: u32) -> u32 {
  let ux = u32(abs(ci.x));
  let uy = u32(abs(ci.y));
  let uz = u32(abs(ci.z));
  let h = (ux * 73856093u) ^ (uy * 19349663u) ^ (uz * 83492791u);
  return h % hashSize;
}

// ── Jetカラーマップ ───────────────────────────────────────────────
fn jetColor(t: f32) -> vec3f {
  let tc = clamp(t, 0.0f, 1.0f);
  var r: f32; var g: f32; var b: f32;
  if      (tc < 0.125f) { r = 0.0f;          g = 0.0f;                   b = 0.5f + 4.0f*tc; }
  else if (tc < 0.375f) { r = 0.0f;          g = 4.0f*(tc-0.125f);       b = 1.0f; }
  else if (tc < 0.625f) { r = 4.0f*(tc-0.375f); g = 1.0f;                b = 1.0f-4.0f*(tc-0.375f); }
  else if (tc < 0.875f) { r = 1.0f;          g = 1.0f-4.0f*(tc-0.625f); b = 0.0f; }
  else                  { r = 1.0f-4.0f*(tc-0.875f); g = 0.0f;           b = 0.0f; }
  return vec3f(r, g, b);
}

fn nozzleInnerRadius(nozzle: u32, zNoz: f32) -> f32 {
  let z = clamp(zNoz, 0.0f, 0.18f);
  if (z <= 0.04f) {
    let bottom = select(0.0425f, select(0.0475f, 0.0350f, nozzle == 4u), nozzle == 3u);
    return bottom + (0.0200f - bottom) * z / 0.04f;
  }
  let top = 0.0550f;
  return 0.0200f + (top - 0.0200f) * (z - 0.04f) / 0.14f;
}

fn nozzleOuterRadius(nozzle: u32, zNoz: f32) -> f32 {
  let z = clamp(zNoz, 0.0f, 0.18f);
  if (z <= 0.04f) {
    let bottom = select(0.0465f, select(0.0515f, 0.0390f, nozzle == 4u), nozzle == 3u);
    return bottom + (0.0240f - bottom) * z / 0.04f;
  }
  return 0.0240f + (0.0590f - 0.0240f) * (z - 0.04f) / 0.14f;
}
`;

// ─────────────────────────────────────────────────────────────────
// ハッシュテーブルリセット
// ─────────────────────────────────────────────────────────────────
const SHADER_HASH_ZERO = /* wgsl */`
${WGSL_COMMON}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> hashCount: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> hashData : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.HASH_SIZE) { return; }
  atomicStore(&hashCount[i], 0u);
  // hashDataも必要に応じてクリア（パフォーマンス最適化: 省略可）
}
`;

// ─────────────────────────────────────────────────────────────────
// ハッシュテーブル構築
// ─────────────────────────────────────────────────────────────────
const SHADER_HASH_BUILD = /* wgsl */`
${WGSL_COMMON}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> hashCount: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> hashData : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.N) { return; }

  let pos  = particles[i].pos.xyz;
  let ci   = vec3i(floor(pos / p.CELL_SIZE));
  let hash = cellHash(ci, p.HASH_SIZE);

  let slot = atomicAdd(&hashCount[hash], 1u);
  if (slot < p.MAX_CELL) {
    hashData[hash * p.MAX_CELL + slot] = i;
  }
}
`;

// ─────────────────────────────────────────────────────────────────
// ハッシュカウントを非アトミックバッファにコピー
// ─────────────────────────────────────────────────────────────────
const SHADER_HASH_COPY = /* wgsl */`
${WGSL_COMMON}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> hashCount    : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> hashCountRead: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.HASH_SIZE) { return; }
  hashCountRead[i] = atomicLoad(&hashCount[i]);
}
`;

// ─────────────────────────────────────────────────────────────────
// MPS数密度計算 + EOS圧力
// particles_in → particles_out (ping-pong)
// ─────────────────────────────────────────────────────────────────
const SHADER_DENSITY = /* wgsl */`
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>               p            : Params;
@group(0) @binding(1) var<storage, read>         particles_in : array<Particle>;
@group(0) @binding(2) var<storage, read_write>   particles_out: array<Particle>;
@group(0) @binding(3) var<storage, read>         hashCountRead: array<u32>;
@group(0) @binding(4) var<storage, read>         hashData     : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.N) { return; }

  var part = particles_in[i];

  // 壁・流入粒子は数密度計算のみ（更新しない）
  if (part.pos.w >= 0.5f) {
    particles_out[i] = part;
    return;
  }

  let pi = part.pos.xyz;
  var ni = 0.0f;

  let ci = vec3i(floor(pi / p.CELL_SIZE));
  for (var dx = -1; dx <= 1; dx++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dz = -1; dz <= 1; dz++) {
    let nc   = ci + vec3i(dx, dy, dz);
    let h    = cellHash(nc, p.HASH_SIZE);
    let cnt  = min(hashCountRead[h], p.MAX_CELL);
    let base = h * p.MAX_CELL;
    for (var k = 0u; k < cnt; k++) {
      let j = hashData[base + k];
      if (j == i) { continue; }
      let pj = particles_in[j];
      let r  = length(pj.pos.xyz - pi);
      // 質量重み付き: 微細粒子(mr小)は体積相当分だけ数密度に寄与する
      ni += mpsWm(r, p.RE, pj.aux2.x);
    }
  }}}

  // EOS圧力: 圧縮率に基づく (負圧はクランプ)
  var press = min(max(0.0f, -p.KAPPA * p.RHO0 / (p.DT * p.DT) * (ni / p.N0 - 1.0f)), 500.0f);

  // ── 床面アウトレット: ゲージ圧=0 (大気圧Dirichlet BC) ─────────
  // 床全面を排気口として扱い、圧力を大気圧基準にリセットする。
  // これにより天井ノズルからの継続的な流入に伴う圧力上昇を防ぐ。
  let floor_outlet_h = p.L0 * 1.5f;
  if (pi.z <= floor_outlet_h) {
    press = 0.0f;
  }

  part.aux.y    = ni;
  part.vel.w    = press;
  particles_out[i] = part;
}
`;

// ─────────────────────────────────────────────────────────────────
// MPS明示的ステップ (粘性 + 圧力勾配 + 重力 + 浮力)
// ─────────────────────────────────────────────────────────────────
const SHADER_EXPLICIT = /* wgsl */`
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>               p            : Params;
@group(0) @binding(1) var<storage, read>         particles_in : array<Particle>;
@group(0) @binding(2) var<storage, read_write>   particles_out: array<Particle>;
@group(0) @binding(3) var<storage, read>         hashCountRead: array<u32>;
@group(0) @binding(4) var<storage, read>         hashData     : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.N) { return; }

  var part = particles_in[i];
  let ptype = part.pos.w;

  // 壁粒子だけを固定。流入粒子はノズル内を実際に移流させる。
  if (ptype > 0.5f && ptype < 1.5f) {
    particles_out[i] = part;
    return;
  }

  // ── ノズル開口直下の運動量注入ゾーン ────────────────────────────
  // ノズル粒子数 (~数十個) が室内粒子数 (~200k) に対して絶対数不足のため、
  // MPS相互作用だけでは噴流の運動量が室内に十分伝わらない。
  // そこで天井から JET_DEPTH 以内 & ノズル内径内の粒子に対して、
  // 下向き速度ターゲットに向けたソフトブレンドを毎ステップ適用し、
  // ノズルの「仮想アクチュエータディスク」として機能させる。
  //   ・天井直下 (frac > 0.7): 速度を強制上書き
  //   ・中間帯 (frac ≤ 0.7): ターゲットへの 10% ブレンド (物理的な慣性を維持)
  //
  // ★ Bug fix: 従来はゾーン内の"全"流体粒子(aux.w=0の周囲室内空気も含む)を
  //   無条件に引き込んで下向きに強制していた。これによりノズル起源ではない
  //   室内空気まで狭い円柱領域に押し込まれて過密になり、SHADER_DENSITYの
  //   EOS圧力(ni/N0超過で急上昇)が跳ね上がって強い水平方向の反発力を生み、
  //   天井直下でT字型に横へ広がる低速の粒子塊として観測されていた。
  //   → 対象をノズル起源の粒子(aux.w>0.5=循環冷気粒子, ptype>1.5=流入BC粒子)
  //     に限定し、さらに局所数密度(aux.y, 前段SHADER_DENSITYで算出済み)が
  //     N0を超えて過密な場合は注入を弱める密度フィードバックを追加する。
  let is_nozzle_origin = (part.aux.w > 0.5f) || (ptype > 1.5f);
  let inlet_dx = part.pos.x - p.IN_CX;
  let inlet_dy = part.pos.y - p.IN_CY;
  let inlet_r = length(vec2f(inlet_dx, inlet_dy));
  // 微細解像度バッファ柱(天井直下0.30m、mr=1/27で高解像度化)の深さに合わせる。
  // この範囲は粒子解像度が十分(Ø直径あたり~7粒子)なので、強制注入は
  // あくまで「噴流の初期形成を助ける」補助であり、範囲外は自然なMPS物理に委ねる。
  let jet_depth = 0.40f;
  let jet_z_bot = p.DOM - jet_depth;
  // 過密度フィードバック: ni/N0 が1.5倍を超えたら注入をゼロまで弱める
  let crowd_ratio = part.aux.y / p.N0;
  let crowd_damp  = clamp(1.5f - crowd_ratio, 0.0f, 1.0f);
  if (is_nozzle_origin && part.pos.z >= jet_z_bot && part.pos.z <= p.DOM && inlet_r <= p.IN_R) {
    let jet_frac = (part.pos.z - jet_z_bot) / jet_depth;  // 0=ゾーン下端, 1=天井
    let v_target = -p.V_MAX / 1.5f;
    if (jet_frac > 0.7f) {
      // 天井直下: 過密でなければ強制上書き、過密なら自然物理側に委ねる比率を増やす
      part.vel.z = mix(part.vel.z, v_target, crowd_damp);
      // 過密ゾーンでの水平方向への弾かれ(T字拡散)を抑える軽い減衰
      part.vel.x *= 0.9f;
      part.vel.y *= 0.9f;
    } else {
      // 中間帯: ソフトブレンド (物理的な慣性を維持)
      part.vel.z = mix(part.vel.z, v_target * jet_frac, 0.10f * crowd_damp);
    }
    part.aux.z = abs(part.vel.z);
  }

  let pi    = part.pos.xyz;
  let vi    = part.vel.xyz;
  let Ti    = part.aux.x;
  let pi_p  = part.vel.w;    // 自粒子圧力

  // MPS演算子係数
  let COEF_LAP  = (2.0f * 3.0f) / (p.N0 * p.LAMBDA);   // ラプラシアン
  let COEF_GRAD = 3.0f / p.N0;                           // 勾配

  var lapV = vec3f(0.0f);    // 速度のラプラシアン (粘性)
  var gradP= vec3f(0.0f);    // 圧力勾配

  let ci = vec3i(floor(pi / p.CELL_SIZE));
  for (var dx = -1; dx <= 1; dx++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dz = -1; dz <= 1; dz++) {
    let nc   = ci + vec3i(dx, dy, dz);
    let h    = cellHash(nc, p.HASH_SIZE);
    let cnt  = min(hashCountRead[h], p.MAX_CELL);
    let base = h * p.MAX_CELL;
    for (var k = 0u; k < cnt; k++) {
      let j = hashData[base + k];
      if (j == i) { continue; }
      let pj  = particles_in[j];
      let rij = pj.pos.xyz - pi;
      let r   = length(rij);
      if (r >= p.RE || r < 1e-7f) { continue; }

      // 質量重み付きカーネル: 微細解像度領域(mr<1)の粒子は
      // 体積比分だけ寄与を割り引き、粗い領域との整合を保つ
      let wij = mpsWm(r, p.RE, pj.aux2.x);

      // ラプラシアン (粘性): Σ(v_j - v_i) * w(r) * mr_j
      lapV += (pj.vel.xyz - vi) * wij;

      // 圧力勾配: Σ (p_i + p_j) / r² * rij * w(r) * mr_j
      // (Koshizuka 1996の対称モデルを質量重み付きに拡張)
      let dp = (pi_p + pj.vel.w) / (r * r);
      gradP += dp * rij * wij;
    }
  }}}

  // 加速度計算
  let a_visc = COEF_LAP  * p.NU * lapV;
  let a_pres = -COEF_GRAD / p.RHO0 * gradP;

  // 浮力 (Boussinesq近似): 冷気は下降(−z), 暖気は上昇(+z)
  let dT    = Ti - p.T_REF;
  let a_buoy= vec3f(0.0f, 0.0f, p.BETA * dT * p.G);

  // 重力 (−z方向)
  let a_grav= vec3f(0.0f, 0.0f, -p.G);

  let a_total = a_visc + a_pres + a_buoy + a_grav;

  // 速度・位置更新
  var v_new = vi + p.DT * a_total;
  var r_new = pi + p.DT * v_new;

  // ── 壁面境界条件 (no-slip) ─────────────────────────────────────
  let eps  = p.L0 * 0.51f;
  let dom  = p.DOM - eps;
  // x, y: 室内側壁 (no-slip)
  if (r_new.x < eps)  { r_new.x = eps;  v_new.x = 0.0f; }
  if (r_new.x > dom)  { r_new.x = dom;  v_new.x = 0.0f; }
  if (r_new.y < eps)  { r_new.y = eps;  v_new.y = 0.0f; }
  if (r_new.y > dom)  { r_new.y = dom;  v_new.y = 0.0f; }
  // z: 床面 = アウトレット (no-slipを解除し、流出した粒子をリサイクル)
  // 粒子が床を通過してz<0になったら、天井直下にアンビエント条件で再投入する。
  // x,yはノズル開口外のランダム位置に分散させ、噴流を乱さない。
  if (r_new.z < 0.0f) {
    // ★ Bug fix: ノズル粒子(aux.w>0.5)は、高速移動により
    //   InletBC側のfloor_eps判定窓(z<=L0*0.55)を1ステップで飛び越えて
    //   直接z<0へ到達することがある。従来はここで無条件にaux.w=0へ
    //   クリアしていたため、SHADER_INLET_BCの再投入処理(nozzle_flag>0.5判定)
    //   が二度と発火せず、初回の吹き出しパルス後に冷気供給が途絶えていた。
    //   → ノズル粒子は位置を床面付近にクランプするだけに留め、
    //     aux.w/温度は保持して、後段のSHADER_INLET_BCに再投入を委ねる。
    if (part.aux.w > 0.5f) {
      r_new.z = 0.0f;
      v_new   = vec3f(0.0f, 0.0f, v_new.z);  // 水平成分のみ止める。zはInletBCが再設定
    } else {
      // ★ Bug fix (再修正): 直前の修正で床を単純なno-slip壁にしたところ、
      //   循環経路が失われ、固定粒子数の閉空間に天井から注入され続けた
      //   結果、床側に粒子が積み重なって圧力(EOS, 上限500Pa)が飽和し、
      //   系全体が身動きできない完全なジャム状態(最大速度0.0m/s)に陥った。
      //   循環経路(床→天井への還流)自体は閉空間で質量収支を保つために
      //   必要だったため復活させるが、以下の点を緩和する:
      //     1) 天井直下の薄い1層(旧: DOM-3*L0固定)に集中させず、
      //        天井から0.3〜1.2mの範囲に高さも分散 → 1箇所の過密を回避
      //     2) 速度を完全ゼロにせず20%残す → 真空へワープした直後の
      //        「重力だけの自由落下」的な不自然な加速を緩和
      //     3) 温度も瞬時にT_REFへ飛ばさず緩やかにブレンド → 浮力の急変
      //        (dT=0による打ち消し)を避ける
      let seed   = u32(i) * 1664525u + u32(p.step) * 1013904223u;
      let angle  = f32(seed % 1024u) / 1024.0f * 6.28318f;
      let r_min  = p.IN_R * 1.4f;
      let r_max  = p.DOM * 0.45f;
      let rnd_r  = r_min + f32((seed >> 10u) % 512u) / 512.0f * (r_max - r_min);
      let z_seed = (seed >> 20u) % 1024u;
      let z_frac = f32(z_seed) / 1024.0f;   // 0〜1
      r_new.x = clamp(p.IN_CX + rnd_r * cos(angle), eps, dom);
      r_new.y = clamp(p.IN_CY + rnd_r * sin(angle), eps, dom);
      r_new.z = p.DOM - (p.L0 * 3.0f + z_frac * 0.9f);  // 天井から0.3〜1.2m下に分散
      v_new      = v_new * 0.2f;                    // 完全停止ではなく大きく減衰のみ
      part.aux.x = mix(part.aux.x, p.T_REF, 0.3f);  // 周囲温度へ緩やかに近づける
    }
  }
  // z: 上限 → 粒子がノズル内流体なら DOM_Z まで、室内粒子なら DOM まで
  // 簡易判定: z>DOM のとき水平距離でノズル内か判定
  let dom_z = p.DOM_Z - eps;
  if (r_new.z > dom_z) { r_new.z = dom_z; v_new.z = 0.0f; }
  // 室内粒子が天井(z>DOM)に上がった場合、ノズル開口外なら押し返す
  // 開口開口判定: IN_R * 1.15 ≈ rOuter（外径）近傼に拡大して十分な流路を確保
  if (r_new.z > p.DOM - eps) {
    let hdx = r_new.x - p.IN_CX;
    let hdy = r_new.y - p.IN_CY;
    let hr  = sqrt(hdx*hdx + hdy*hdy);
    let open_r = p.IN_R * 1.15f;  // 外径相当 (rOuter ≈ rInner * 1.1)
    if (hr > open_r) {
      // ノズル開口外 → 天井に当たる
      r_new.z = p.DOM - eps;
      v_new.z = 0.0f;
    }
  }

  // ノズル内壁と内部シリンダーのno-slip近似。粒子が壁を横切らないようにする。
  if (r_new.z > p.DOM && r_new.z < p.DOM_Z) {
    let z_nozzle = r_new.z - p.DOM;
    let dx_nozzle = r_new.x - p.IN_CX;
    let dy_nozzle = r_new.y - p.IN_CY;
    let radial = sqrt(dx_nozzle * dx_nozzle + dy_nozzle * dy_nozzle);
    // 内壁発効半径: ノズル内径にここでも外径相当の忤を使う
    let inner_r = nozzleInnerRadius(p.NOZZLE, z_nozzle);
    let outer_r = nozzleOuterRadius(p.NOZZLE, z_nozzle);
    let wall_radius = (inner_r + outer_r) * 0.5f - 0.003f;  // 内径と外径の中間
    if (radial > wall_radius) {
      let safe_radius = max(wall_radius, 0.001f);
      let scale = safe_radius / max(radial, 0.0001f);
      r_new.x = p.IN_CX + dx_nozzle * scale;
      r_new.y = p.IN_CY + dy_nozzle * scale;
      let radial_velocity = (v_new.x * dx_nozzle + v_new.y * dy_nozzle) / max(radial, 0.0001f);
      v_new.x -= radial_velocity * dx_nozzle / max(radial, 0.0001f);
      v_new.y -= radial_velocity * dy_nozzle / max(radial, 0.0001f);
    }
    if (p.NOZZLE == 2u && z_nozzle >= 0.04f) {
      let core_radius = p.CORE_R + 0.003f;
      if (radial < core_radius) {
        let safe_radius = max(core_radius, 0.001f);
        let scale = safe_radius / max(radial, 0.0001f);
        r_new.x = p.IN_CX + dx_nozzle * scale;
        r_new.y = p.IN_CY + dy_nozzle * scale;
      }
    }
  }

  // 速度上限（数値発散防止）
  let vmax_clip = 20.0f;
  let vlen = length(v_new);
  if (vlen > vmax_clip) { v_new = v_new * (vmax_clip / vlen); }

  part.pos  = vec4f(r_new, ptype);
  part.vel  = vec4f(v_new, pi_p);
  part.aux.z= length(v_new);   // 速度大きさ (表示用)
  particles_out[i] = part;
}
`;

// ─────────────────────────────────────────────────────────────────
// 温度輸送 (熱拡散 + 流入粒子のDirichlet条件)
// ─────────────────────────────────────────────────────────────────
const SHADER_TEMPERATURE = /* wgsl */`
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>               p            : Params;
@group(0) @binding(1) var<storage, read>         particles_in : array<Particle>;
@group(0) @binding(2) var<storage, read_write>   particles_out: array<Particle>;
@group(0) @binding(3) var<storage, read>         hashCountRead: array<u32>;
@group(0) @binding(4) var<storage, read>         hashData     : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.N) { return; }

  var part  = particles_in[i];
  let ptype = part.pos.w;

  // 流入粒子: 温度固定
  if (ptype > 1.5f) {
    part.aux.x = p.T_IN;
    particles_out[i] = part;
    return;
  }
  // 壁面粒子: 断熱（温度変化なし）
  if (ptype > 0.5f) {
    particles_out[i] = part;
    return;
  }

  let inlet_dx = part.pos.x - p.IN_CX;
  let inlet_dy = part.pos.y - p.IN_CY;
  let inlet_r = length(vec2f(inlet_dx, inlet_dy));
  // ノズル直下 3L0 以内: Dirichlet 温度境界条件
  // (運動量注入ゾーン10L0より狭く設定し、下部は自然な熱拡散・移流に委ねる)
  if (part.pos.z >= p.DOM - p.L0 * 3.0f && part.pos.z <= p.DOM && inlet_r <= p.IN_R) {
    part.aux.x = p.T_IN;
    particles_out[i] = part;
    return;
  }

  // ノズル吹き出し温度は23.5℃で一定。部屋側だけ熱拡散を計算する。
  if (part.pos.z > p.DOM) {
    part.aux.x = p.T_IN;
    particles_out[i] = part;
    return;
  }

  let pi = part.pos.xyz;
  let Ti = part.aux.x;
  var lapT = 0.0f;

  let COEF_LAP = (2.0f * 3.0f) / (p.N0 * p.LAMBDA);

  let ci = vec3i(floor(pi / p.CELL_SIZE));
  for (var dx = -1; dx <= 1; dx++) {
  for (var dy = -1; dy <= 1; dy++) {
  for (var dz = -1; dz <= 1; dz++) {
    let nc   = ci + vec3i(dx, dy, dz);
    let h    = cellHash(nc, p.HASH_SIZE);
    let cnt  = min(hashCountRead[h], p.MAX_CELL);
    let base = h * p.MAX_CELL;
    for (var k = 0u; k < cnt; k++) {
      let j = hashData[base + k];
      if (j == i) { continue; }
      let pj = particles_in[j];
      let r  = length(pj.pos.xyz - pi);
      if (r >= p.RE || r < 1e-7f) { continue; }
      lapT += (pj.aux.x - Ti) * mpsWm(r, p.RE, pj.aux2.x);
    }
  }}}

  // 温度更新 (陽解法熱拡散)
  var T_new = Ti + p.DT * p.ALPHA * COEF_LAP * lapT;
  T_new = clamp(T_new, p.T_IN - 1.0f, p.T_REF + 2.0f);

  part.aux.x = T_new;
  particles_out[i] = part;
}
`;

// ─────────────────────────────────────────────────────────────────
// 流入境界条件の再印加
//
// 設計:
//  ・ノズル上面 (z=DOM_Z) が物理的な流入面 (空調接続口)
//  ・空気はノズル内を下降し、天井開口 (z=DOM) から室内に吹き出す
//  ・ptype=2 粒子は全ノズル領域 (z: DOM〜DOM_Z) に常駐する固定BC
//  ・z<=DOMに出た場合はノズル上面に再投入 → 継続的噴流を実現
//  ・MPS粘性・圧力を介して室内粒子(ptype=0)への運動量伝達は維持
// ─────────────────────────────────────────────────────────────────
const SHADER_INLET_BC = /* wgsl */`
${WGSL_COMMON}
@group(0) @binding(0) var<uniform>             p        : Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.N) { return; }

  var part = particles[i];
  let ptype = part.pos.w;

  // ── 壁面粒子 (ptype=1): スキップ ─────────────────────────────────
  if (ptype > 0.5f && ptype < 1.5f) { return; }

  // ── 流入BC粒子 (ptype=2): ノズル内常駐 + 壁面角度に沿った速度強制 ──
  if (ptype > 1.5f) {
    let dr_center = vec2f(part.pos.x - p.IN_CX, part.pos.y - p.IN_CY);
    let dr_len = length(dr_center);

    // ── テレポート: ノズル出口を通過した粒子をノズル上面に再投入 ────
    // 物理的意味: ノズル上面から常に新鮮な空気が供給される
    if (part.pos.z <= p.DOM) {
      part.pos.z = p.DOM + (p.DOM_Z - p.DOM) * 0.92f;
    }

    // ── 断面積保存則による軸方向速度 ────────────────────────────────
    let z_nozzle = clamp(part.pos.z - p.DOM, 0.0f, 0.18f);
    let nozzle_r = nozzleInnerRadius(p.NOZZLE, z_nozzle);
    let core_r   = select(0.0f, p.CORE_R, p.NOZZLE == 2u && z_nozzle >= 0.04f);
    let flow_area = max(nozzle_r * nozzle_r - core_r * core_r, 0.0004f);
    // V_in * A_in = v_axial * A_local  (連続の式)
    let v_axial = min(p.V_IN * 0.055f * 0.055f / flow_area, 20.0f);

    // ── 壁面角度: 有限差分で dR/dz を計算 (z上向き正) ────────────
    // プロファイルから局所的な内壁傾斜を数値微分
    let dz_fd   = 0.003f;
    let r_above = nozzleInnerRadius(p.NOZZLE, min(z_nozzle + dz_fd, 0.18f));
    let dr_dz   = (r_above - nozzle_r) / dz_fd;
    // 下向き流れで径方向に広がる条件: dr_dz < 0 (z低下→r増加)
    // → 発散部 (出口付近) のみ外向き成分を与える
    let spread_tan = max(0.0f, -dr_dz);  // 広がり角のtangent (≥0)

    // ── 速度ベクトルを壁面角度方向に分解 ─────────────────────────
    // 速さ v_axial を軸方向と径方向に分配
    // |v| = v_axial, vz = -v_axial*cos(α), vr = v_axial*sin(α)
    // tan(α) = spread_tan → cos = 1/√(1+tan²), sin = tan/√(1+tan²)
    let angle_mag = sqrt(1.0f + spread_tan * spread_tan);
    let vz_comp   = -v_axial / angle_mag;         // 軸方向 (下向き)
    let vr_comp   =  v_axial * spread_tan / angle_mag;  // 径方向 (外向き)

    // 径方向の単位ベクトルで vx, vy に分解
    // 中心粒子 (r≈0) は径方向不定 → vx=vy=0 のみ
    var vx_out = 0.0f;
    var vy_out = 0.0f;
    if (dr_len > 1e-4f) {
      vx_out = vr_comp * dr_center.x / dr_len;
      vy_out = vr_comp * dr_center.y / dr_len;
    }

    // ── ノズル2 内部シリンダー中心部: 速度ゼロ ─────────────────────
    let v_scale = select(1.0f, 0.0f,
                         p.NOZZLE == 2u && p.HAS_CORE == 1u && dr_len < p.CORE_R);

    part.vel   = vec4f(vx_out * v_scale, vy_out * v_scale,
                       vz_comp * v_scale, part.vel.w);
    part.aux.x = p.T_IN;
    part.aux.z = length(vec3f(vx_out, vy_out, vz_comp)) * v_scale;
    particles[i] = part;
    return;
  }

  // ── 流体粒子 (ptype=0) ────────────────────────────────────────────
  // aux.w がノズル粒子フラグ: 1.0 = ノズル粒子, 0.0 = 室内粒子
  let nozzle_flag = part.aux.w;

  // ── ノズル粒子の再投入: 床壁面epsilonに到達したらノズル上面へリサイクル ──
  // 再投入閾値を壁面BC(eps = L0*0.51)に合わせる。
  // これにより床付近で速度を持った粒子が「床から湧く」ように見える問題を解消する。
  // (旧: L0*3.0f は壁面BC到達前に再投入が起き、視覚的に「床アウトレット」に見えた)
  let floor_eps = p.L0 * 0.55f;  // 壁面eps(0.51)より僅かに大きく設定
  if (nozzle_flag > 0.5f && part.pos.z <= floor_eps) {
    // ── ノズル粒子が床壁面に到達 → ノズル上面にリサイクル ───────────
    // x,y をノズル中心近傍にリセット: 開口外に出現して「床アウトレット」に見えるのを防ぐ
    // ランダム径方向成分で円形噴流断面を模擬
    let r_reset  = p.IN_R * 0.6f;  // ノズル内径の60%以内にランダム配置
    // 疑似ランダム: 粒子IDと現在ステップを組み合わせたLCG
    let seed     = u32(i) * 1664525u + u32(p.step) * 1013904223u;
    let angle    = f32(seed % 1024u) / 1024.0f * 6.28318f;
    let rnd_r    = f32((seed >> 10u) % 512u) / 512.0f * r_reset;
    part.pos.x = p.IN_CX + rnd_r * cos(angle);
    part.pos.y = p.IN_CY + rnd_r * sin(angle);
    part.pos.z = p.DOM + (p.DOM_Z - p.DOM) * 0.92f;

    // ノズル上面 (z≈DOM_Z) での入口速度を設定
    let r_top    = nozzleInnerRadius(p.NOZZLE, 0.17f);
    let c_top    = select(0.0f, p.CORE_R, p.NOZZLE == 2u);
    let area_top = max(r_top * r_top - c_top * c_top, 0.0004f);
    let v_top    = min(p.V_IN * 0.055f * 0.055f / area_top, 20.0f);

    part.vel   = vec4f(0.0f, 0.0f, -v_top, part.vel.w);
    part.aux.x = p.T_IN;
    part.aux.z = v_top;
    // aux.w (ノズルフラグ) は維持したまま
    particles[i] = part;
  }
  // 室内粒子 (nozzle_flag=0) は何もしない: MPS物理に任せる
}
`;


// ─────────────────────────────────────────────────────────────────
// 粒子ソート 1/2: カメラ距離キーの初期化
// 半透明ビルボードを正しく「遠→近」の順で描画するための下準備。
// paddedN(2の冪)まではダミーキー(+∞)で埋め、ソート後に末尾へ押しやる。
// ─────────────────────────────────────────────────────────────────
const SHADER_SORT_INIT = /* wgsl */`
struct Particle {
  pos : vec4f, vel : vec4f, aux : vec4f, aux2: vec4f,
};
struct SortParams {
  n       : u32,   // 実粒子数
  paddedN : u32,   // 2の冪に切り上げた配列長
  k       : u32,   // (未使用: init時は0)
  j       : u32,   // (未使用: init時は0)
};
struct Camera {
  mvp: mat4x4f, viewPos: vec4f, slice_mode: f32, slice_pos: f32, slice_thick: f32, view_mode: f32,
};

@group(0) @binding(0) var<uniform> sp: SortParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<uniform> cam: Camera;
@group(0) @binding(3) var<storage, read_write> sortKey: array<f32>;
@group(0) @binding(4) var<storage, read_write> sortIdx: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sp.paddedN) { return; }

  if (i < sp.n) {
    // 物理座標 → Three.js世界座標 (SHADER_VERTのworld計算と同じ変換)
    let pos   = particles[i].pos.xyz;
    let world = vec3f(pos.x - 1.0f, pos.z - 1.0f, pos.y - 1.0f);
    let d     = distance(cam.viewPos.xyz, world);
    sortKey[i] = -d;           // 昇順ソートで「遠い(負に大きい)→近い」の順になる
    sortIdx[i] = i;
  } else {
    sortKey[i] = 1.0e30f;      // パディングは必ず配列末尾(N以降)に押しやる
    sortIdx[i] = 0xffffffffu;
  }
}
`;

// ─────────────────────────────────────────────────────────────────
// 粒子ソート 2/2: バイトニックソート compare-exchange ステップ
// (k,j)の組ごとに1回ディスパッチする。呼び出し側でk,jをスイープする。
// ─────────────────────────────────────────────────────────────────
const SHADER_SORT_STEP = /* wgsl */`
struct SortParams {
  n       : u32,
  paddedN : u32,
  k       : u32,
  j       : u32,
};
@group(0) @binding(0) var<uniform> sp: SortParams;
@group(0) @binding(1) var<storage, read_write> sortKey: array<f32>;
@group(0) @binding(2) var<storage, read_write> sortIdx: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sp.paddedN) { return; }

  let ixj = i ^ sp.j;
  if (ixj <= i || ixj >= sp.paddedN) { return; }  // ペアの片方だけが処理する

  let ascending = (i & sp.k) == 0u;
  let ki = sortKey[i];
  let kj = sortKey[ixj];
  let doSwap = select(ki < kj, (ki > kj), ascending);
  if (doSwap) {
    sortKey[i]   = kj;  sortKey[ixj]   = ki;
    let tmp      = sortIdx[i];
    sortIdx[i]   = sortIdx[ixj];
    sortIdx[ixj] = tmp;
  }
}
`;

// ─────────────────────────────────────────────────────────────────
// レンダリング: 頂点シェーダー
// インスタンス描画: 1パーティクル = 4頂点のビルボードクワッド
// ─────────────────────────────────────────────────────────────────
const SHADER_VERT = /* wgsl */`
${WGSL_COMMON}

struct Camera {
  mvp         : mat4x4f,
  viewPos     : vec4f,
  slice_mode  : f32,    // 0=全体表示, 1=YZ断面(x固定), 2=XZ断面(y固定)
  slice_pos   : f32,    // 断面位置 [物理m]
  slice_thick : f32,    // 断面半幅 [m]
  view_mode   : f32,    // 0=3D粒子, 1=断面, 2=コンター
};

@group(0) @binding(0) var<uniform> p      : Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<uniform> cam    : Camera;
@group(0) @binding(3) var<storage, read> sortIdx: array<u32>;  // カメラ距離ソート済みインデックス (遠→近)

struct VertOut {
  @builtin(position) pos : vec4f,
  @location(0) color     : vec3f,
  @location(1) uv        : vec2f,
  @location(2) alpha     : f32,
};

// クワッドオフセット (billboard)
const QUAD = array<vec2f, 4>(
  vec2f(-1.0f,  1.0f),
  vec2f( 1.0f,  1.0f),
  vec2f(-1.0f, -1.0f),
  vec2f( 1.0f, -1.0f),
);

@vertex
fn vs_main(
  @builtin(vertex_index)   vid: u32,
  @builtin(instance_index) iid: u32,
) -> VertOut {
  // instance_index はソート後の並び順 (0=最も遠い, N-1=最も近い)。
  // sortIdxで元の粒子配列インデックスに変換してから読む。
  let idx   = sortIdx[iid];
  let part  = particles[idx];
  let ptype = part.pos.w;

  // 壁面粒子は非表示（極小）
  let is_wall = ptype > 0.5f && ptype < 1.5f;

  // 物理座標 → Three.js座標系 (Y=上)
  // 物理: x=左右, y=奥行, z=上下  →  THREE: x=左右, y=上下, z=奥行
  // 室内中心 (1.0, 1.0, 1.0) を原点に平行移動
  // ノズル領域(z>2.0)は [0, 0.18] 上に出る
  let world = vec3f(part.pos.x - 1.0f, part.pos.z - 1.0f, part.pos.y - 1.0f);

  // ── 断面フィルター ─────────────────────────────────────────────
  var slice_alpha = 1.0f;
  if (cam.slice_mode > 0.5f && cam.slice_mode < 1.5f) {
    // YZ断面: x固定 (Three.js世界座標 x = 物理x - 1.0)
    let dist = abs(part.pos.x - cam.slice_pos);
    if (dist > cam.slice_thick) { slice_alpha = 0.0f; }
  } else if (cam.slice_mode > 1.5f) {
    // XZ断面: y固定 (Three.js世界座標 z = 物理y - 1.0)
    let dist = abs(part.pos.y - cam.slice_pos);
    if (dist > cam.slice_thick) { slice_alpha = 0.0f; }
  }

  // クリップ座標
  let clip = cam.mvp * vec4f(world, 1.0f);

  // ビルボードサイズ (NDC単位)
  // visualMode=1 (断面): 3Dモードと同じサイズ (0.004f)
  // visualMode=0 (3D):   通常サイズ (0.004f)
  // visualMode=2 (コンター): 大きめ
  var ptSize = select(0.004f, 0.0f, is_wall);
  if (cam.view_mode > 1.5f) {
    // コンターモード: 背景用に少し大きめ
    ptSize = select(0.018f, 0.0f, is_wall);
  }
  // 流入粒子は少し大きく
  if (ptype > 1.5f) { ptSize = 0.005f; }

  let offset = QUAD[vid] * ptSize;
  let finalPos = vec4f(clip.xy + offset * clip.w, clip.z, clip.w);

  // 表示モードに応じた色計算
  var t: f32;
  if (p.DISP_MODE == 0u) {
    // 温度モード
    t = (part.aux.x - p.T_MIN) / (p.T_MAX - p.T_MIN);
  } else if (p.DISP_MODE == 1u) {
    // 速度モード
    t = part.aux.z / p.V_MAX;
  } else {
    // 圧力モード
    t = clamp(part.vel.w / 500.0f, 0.0f, 1.0f);
  }

  let color = jetColor(clamp(t, 0.0f, 1.0f));
  let alpha = select(1.0f, 0.0f, is_wall) * slice_alpha;

  var out: VertOut;
  out.pos   = finalPos;
  out.color = color;
  out.uv    = QUAD[vid];
  out.alpha = alpha;
  return out;
}
`;

// ─────────────────────────────────────────────────────────────────
// レンダリング: フラグメントシェーダー
// ─────────────────────────────────────────────────────────────────
const SHADER_FRAG = /* wgsl */`
struct VertOut {
  @builtin(position) pos : vec4f,
  @location(0) color     : vec3f,
  @location(1) uv        : vec2f,
  @location(2) alpha     : f32,
};

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
  // 円形クリッピング（クワッドを円に）
  let d = length(in.uv);
  if (d > 1.0f || in.alpha < 0.01f) { discard; }

  // ソフトエッジ
  let edge  = 1.0f - smoothstep(0.6f, 1.0f, d);
  // ライティング効果 (球面法線シミュレーション)
  let nz    = sqrt(max(0.0f, 1.0f - dot(in.uv, in.uv)));
  let light = 0.5f + 0.5f * nz;

  return vec4f(in.color * light, edge * in.alpha);
}
`;

const SHADER_VECTOR_VERT = /* wgsl */`
struct Params { N:u32, NX:u32, step:u32, _pad:u32, H:f32, _a:f32, _b:f32, _c:f32, _l:f32, DT:f32, NU:f32, ALPHA:f32, G:f32, BETA:f32, T_REF:f32, T_IN:f32, RHO0:f32, _k:f32, _cell:f32, DOM:f32, V_IN:f32, NOZZLE:u32, IN_R:f32, IN_CX:f32, IN_CY:f32, HAS_CORE:u32, CORE_R:f32, T_MIN:f32, T_MAX:f32, V_MAX:f32, DISP_MODE:u32, DOM_Z:f32, };
struct Cell { pos:vec4f, vel:vec4f, aux:vec4f, aux2:vec4f, };
struct Camera { mvp:mat4x4f, viewPos:vec4f, slice_mode:f32, slice_pos:f32, slice_thick:f32, view_mode:f32, };
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> cells:array<Cell>;
@group(0) @binding(2) var<uniform> cam:Camera;
struct Out { @builtin(position) pos:vec4f, @location(0) alpha:f32, @location(1) color:vec3f, };

fn jetColor(t: f32) -> vec3f {
  let tc = clamp(t, 0.0f, 1.0f);
  var r: f32; var g: f32; var b: f32;
  if      (tc < 0.125f) { r = 0.0f;             g = 0.0f;                      b = 0.5f + 4.0f * tc; }
  else if (tc < 0.375f) { r = 0.0f;             g = 4.0f * (tc - 0.125f);       b = 1.0f; }
  else if (tc < 0.625f) { r = 4.0f * (tc - 0.375f); g = 1.0f;                   b = 1.0f - 4.0f * (tc - 0.375f); }
  else if (tc < 0.875f) { r = 1.0f;             g = 1.0f - 4.0f * (tc - 0.625f); b = 0.0f; }
  else                  { r = 1.0f - 4.0f * (tc - 0.875f); g = 0.0f;              b = 0.0f; }
  return vec3f(r, g, b);
}

@vertex fn main(@builtin(vertex_index) vi:u32,@builtin(instance_index) ci:u32)->Out {
  let c=cells[ci];
  let wall=c.pos.w>0.5f&&c.pos.w<1.5f;
  let isNozzleCell=c.pos.z>p.DOM;
  let v=c.vel.xyz;
  let speed=length(v);

  // ── 形状パラメータ (cpu_fallback.js _updateVectorLines と同一式) ──────
  //   軸対応: THREE.X=物理x, THREE.Y=物理z(鉛直), THREE.Z=物理y
  //   長さ   len  = clamp(speed*0.007, 0.025, 0.085)
  //   矢頭   hLen = len*0.35、矢頭の半幅 = hLen*0.55
  //   非表示 speed < 0.08
  let start = vec3f(c.pos.x-1.0f, c.pos.z-1.0f, c.pos.y-1.0f);
  let invS  = 1.0f / max(speed, 1e-6f);
  let dir   = vec3f(v.x*invS, v.z*invS, v.y*invS);
  let len   = clamp(speed*0.007f, 0.025f, 0.085f);
  let end   = start + dir*len;
  let hLen  = len*0.35f;

  // ほぼ鉛直な矢印では基準ベクトルを切り替える(CPU版と同一のフォールバック)
  var up = vec3f(0.0f,1.0f,0.0f);
  if (abs(dir.y) > 0.92f) { up = vec3f(1.0f,0.0f,0.0f); }
  var side = cross(dir, up);
  let sideLen = max(length(side), 1e-6f);
  side = (side/sideLen) * (hLen*0.55f);

  let headA = end - dir*hLen + side;
  let headB = end - dir*hLen - side;

  // 3線分(本体・矢頭2本) × 2頂点 = 6頂点/インスタンス
  let segment=vi/2u; let endpoint=vi%2u;
  var a=start; var b=end;
  if (segment==1u) { a=end; b=headA; } else if (segment==2u) { a=end; b=headB; }
  let world = select(a,b,endpoint==1u);

  // ── 間引き・断面選択 ──────────────────────────────────────────────
  // 室内格子セル(ci < roomN):
  //   - 床(k=0), 天井(k=nx-1), 側壁(i,j=0,nx-1) の境界セルは除外 (CPU版と同様に流体内部のみ)
  //   - 断面表示時(isSlice): CPU版(cpu_fallback.js)と同様に、slicePositionを含む「厳密に1つのセル層」
  //     (YZ断面なら i==fIdx, XZ断面なら j==fIdx) だけを表示。
  //     以前は曖昧な厚み判定(25mm)だったため、中心付近(幅3.6mm)で前後7層分のベクトルが
  //     同一画面位置に重なって「1つのセルから傘状に複数ベクトルが表示される」不具合が発生していた。
  //   - 3D表示時(!isSlice): 2セルおきに間引き
  let nx    = p.NX;
  let roomN = nx*nx*nx;
  let isSlice = cam.view_mode > 0.5f;
  var show = true;
  if (ci < roomN) {
    let k=ci/(nx*nx); let rem=ci%(nx*nx); let j=rem/nx; let i=rem%nx;
    if (i==0u || i>=nx-1u || j==0u || j>=nx-1u || k==0u || k>=nx-1u) {
      show = false;
    } else if (isSlice && cam.slice_mode > 0.5f) {
      let fIdx = u32(round(cam.viewPos.w));
      let yz = cam.slice_mode < 1.5f;
      let mySliceIdx = select(j, i, yz);
      if (mySliceIdx != fIdx) { show = false; }
    } else if (!isSlice) {
      if (i%2u!=1u || j%2u!=1u || k%2u!=1u) { show = false; }
    }
  } else {
    show = false; // ノズル微細セルはベクトルを描画しない
  }

  var alpha=select(1.0f,0.0f,wall||p.DISP_MODE!=1u||speed<0.08f||!show||isNozzleCell);

  // 流速スカラー量をJetカラーマップでベクトルの色に反映 (CPU版と同一式)
  let t = clamp(speed / 15.0f, 0.0f, 1.0f);
  let col = jetColor(t);
  var o:Out;o.pos=cam.mvp*vec4f(world,1.0f);o.alpha=alpha;o.color=col;return o;
}
`;
const SHADER_VECTOR_FRAG = /* wgsl */`
@fragment fn main(@location(0) alpha:f32, @location(1) color:vec3f)->@location(0) vec4f {
  if(alpha<0.01f){discard;}
  return vec4f(color, alpha);
}
`;

const SHADER_SLICE_COMPUTE = /* wgsl */`
${WGSL_COMMON}
struct Camera {
  mvp: mat4x4f, viewPos: vec4f, slice_mode: f32, slice_pos: f32, slice_thick: f32, view_mode: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> hashCount: array<u32>;
@group(0) @binding(3) var<storage, read> hashData: array<u32>;
@group(0) @binding(4) var<uniform> cam: Camera;
@group(0) @binding(5) var<storage, read_write> values: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= 65u || gid.y >= 65u) { return; }
  let index = gid.y * 65u + gid.x;
  let u = f32(gid.x) / 64.0f * p.DOM;
  let v = f32(gid.y) / 64.0f * p.DOM_Z;
  let plane = cam.slice_pos;
  var sample = vec3f(plane, u, v);
  if (cam.slice_mode > 1.5f) { sample = vec3f(u, plane, v); }
  let radial = select(abs(sample.y - p.DOM / 2.0f), abs(sample.x - p.DOM / 2.0f), cam.slice_mode > 1.5f);
  if (v > p.DOM && radial > nozzleInnerRadius(p.NOZZLE, v - p.DOM)) {
    values[index] = -1.0f;
    return;
  }
  if (p.NOZZLE == 2u && v > p.DOM + 0.04f && radial < p.CORE_R) {
    values[index] = -1.0f;
    return;
  }
  if (v > p.DOM) {
    if (p.DISP_MODE == 0u) { values[index] = 0.0f; }
    else if (p.DISP_MODE == 1u) { values[index] = 1.0f / 1.5f; }
    else { values[index] = 0.0f; }
    return;
  }
  let cell = vec3i(floor(sample / p.CELL_SIZE));
  var weighted = 0.0f;
  var weightSum = 0.0f;
  // ±2セル探索 (セルサイズ=RE=0.063m → 実効探索半径 ~0.2m)
  // ノイズの多い圧力場等を十分な数の粒子で平均化するため範囲を拡大
  for (var dx = -2; dx <= 2; dx++) {
    for (var dy = -2; dy <= 2; dy++) {
      for (var dz = -2; dz <= 2; dz++) {
        let h = cellHash(cell + vec3i(dx, dy, dz), p.HASH_SIZE);
        let count = min(hashCount[h], p.MAX_CELL);
        for (var k = 0u; k < count; k++) {
          let part = particles[hashData[h * p.MAX_CELL + k]];
          if (part.pos.w > 0.5f && part.pos.w < 1.5f) { continue; }
          let d = part.pos.xyz - sample;
          let distance2 = dot(d, d);
          if (distance2 > 0.04f || (abs(d.x) > cam.slice_thick && cam.slice_mode < 1.5f) || (abs(d.y) > cam.slice_thick && cam.slice_mode > 1.5f)) { continue; }
          let w = 1.0f / max(distance2, 0.01f);
          var value = part.aux.x;
          if (p.DISP_MODE == 1u) { value = part.aux.z; }
          if (p.DISP_MODE == 2u) { value = abs(part.vel.w); }
          weighted += value * w;
          weightSum += w;
        }
      }
    }
  }
  var value = select(select(0.0f, p.T_REF, p.DISP_MODE == 0u), weighted / max(weightSum, 0.0001f), weightSum > 0.0f);
  if (p.DISP_MODE == 0u) { value = (value - p.T_MIN) / (p.T_MAX - p.T_MIN); }
  else if (p.DISP_MODE == 1u) { value = value / p.V_MAX; }
  else { value = value / 500.0f; }
  values[index] = clamp(value, 0.0f, 1.0f);
}
`;

const SHADER_SLICE_VERT = /* wgsl */`
struct Camera { mvp: mat4x4f, viewPos: vec4f, slice_mode: f32, slice_pos: f32, slice_thick: f32, view_mode: f32, };
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> values: array<f32>;
struct Out { @builtin(position) pos: vec4f, @location(0) value: f32, @location(1) world: vec3f, };
@vertex fn main(@builtin(vertex_index) id: u32) -> Out {
  let cell = id / 6u; let corner = id % 6u;
  let col = cell % 64u; let row = cell / 64u;
  let c = array<vec2f, 6>(vec2f(0,0),vec2f(1,0),vec2f(0,1),vec2f(1,0),vec2f(1,1),vec2f(0,1))[corner];
  let gx = f32(col) + c.x; let gy = f32(row) + c.y;
  // コンター面は部屋だけ(床 z=0 〜 天井 z=DOM)を描画する。
  // ノズルはこの面の上に独立したワイヤーフレームとして表示する。
  let u = gx / 64.0f * 2.0f; let v = gy / 64.0f * 2.0f;
  var world = vec3f(u - 1.0f, v - 1.0f, cam.slice_pos - 1.0f);
  if (cam.slice_mode < 1.5f) { world = vec3f(cam.slice_pos - 1.0f, v - 1.0f, u - 1.0f); }
  let vi = u32(gy) * 65u + u32(gx);
  var out: Out; out.pos = cam.mvp * vec4f(world, 1.0f); out.value = values[vi]; out.world = world; return out;
}
`;

const SHADER_SLICE_FRAG = /* wgsl */`
${WGSL_COMMON}
@group(0) @binding(2) var<uniform> p: Params;
struct SliceCamera { mvp: mat4x4f, viewPos: vec4f, slice_mode: f32, slice_pos: f32, slice_thick: f32, view_mode: f32, };
@group(0) @binding(3) var<uniform> cam: SliceCamera;
@fragment fn main(@location(0) value: f32, @location(1) world: vec3f) -> @location(0) vec4f {
  let z_nozzle = world.y + 1.0f - p.DOM;
  if (z_nozzle >= 0.0f && z_nozzle <= 0.18f) {
    let sectionRadial = select(abs(world.z), abs(world.x), cam.slice_mode > 1.5f);
    let r = nozzleInnerRadius(p.NOZZLE, z_nozzle);
    let outer = nozzleOuterRadius(p.NOZZLE, z_nozzle);
    if (abs(sectionRadial - outer) < 0.0035f) { return vec4f(0.3f, 0.85f, 1.0f, 1.0f); }
    if (abs(sectionRadial - r) < 0.0025f) { return vec4f(1.0f, 0.75f, 0.25f, 1.0f); }
  }
  if (value < -0.5f) { discard; }
  return vec4f(jetColor(value), 0.78f);
}
`;

// ─────────────────────────────────────────────────────────────────
// 部屋ワイヤーフレーム + ノズルメッシュ用シェーダー
// ─────────────────────────────────────────────────────────────────
const SHADER_ROOM_VERT = /* wgsl */`
struct Camera {
  mvp     : mat4x4f,
  viewPos : vec4f,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct RoomVert {
  @builtin(position) pos: vec4f,
  @location(0) world    : vec3f,
};

@vertex
fn vs_room(
  @location(0) position: vec3f,
) -> RoomVert {
  var out: RoomVert;
  out.pos   = cam.mvp * vec4f(position, 1.0f);
  out.world = position;
  return out;
}
`;

const SHADER_ROOM_FRAG = /* wgsl */`
struct RoomVert {
  @builtin(position) pos: vec4f,
  @location(0) world    : vec3f,
};

@fragment
fn fs_room(in: RoomVert) -> @location(0) vec4f {
  return vec4f(0.4f, 0.5f, 0.7f, 0.35f);
}
`;

const SHADER_NOZZLE_VERT = /* wgsl */`
struct Camera { mvp: mat4x4f, viewPos: vec4f, };
@group(0) @binding(0) var<uniform> cam: Camera;
struct Out { @builtin(position) pos: vec4f, };
@vertex fn main(@location(0) position: vec3f) -> Out {
  var out: Out;
  out.pos = cam.mvp * vec4f(position, 1.0f);
  return out;
}
`;

const SHADER_NOZZLE_FRAG = /* wgsl */`
@fragment fn main() -> @location(0) vec4f {
  return vec4f(0.25f, 0.80f, 1.0f, 0.85f);
}
`;

// ─────────────────────────────────────────────────────────────────
// 有限体積法(FVM): セル中心Cartesian格子用シェーダー
// ─────────────────────────────────────────────────────────────────
const FVM_COMMON = /* wgsl */`
struct Params {
  N:u32, NX:u32, step:u32, _pad:u32,
  H:f32, _r1:f32, _r2:f32, _r3:f32,
  _l:f32, DT:f32, NU:f32, ALPHA:f32,
  G:f32, BETA:f32, T_REF:f32, T_IN:f32,
  RHO0:f32, _k:f32, _cell:f32, DOM:f32,
  V_IN:f32, NOZZLE:u32, IN_R:f32, IN_CX:f32,
  IN_CY:f32, HAS_CORE:u32, CORE_R:f32, T_MIN:f32,
  T_MAX:f32, V_MAX:f32, DISP_MODE:u32, DOM_Z:f32,
};
struct Cell {
  pos:vec4f, vel:vec4f, aux:vec4f, aux2:vec4f,
};
fn index3(i:i32,j:i32,k:i32,n:u32)->u32 {
  let x=clamp(i,0,i32(n)-1); let y=clamp(j,0,i32(n)-1); let z=clamp(k,0,i32(n)-1);
  return u32(x)+n*(u32(y)+n*u32(z));
}
fn cellCoord(c:u32,n:u32)->vec3i {
  let nn=n*n; let k=i32(c/nn); let r=c%nn; let j=i32(r/n); let i=i32(r%n);
  return vec3i(i,j,k);
}
fn nozzleInnerRadius(nozzle:u32,zNoz:f32)->f32 {
  let z=clamp(zNoz,0.0f,0.18f);
  if(z<=0.04f){let bottom=select(0.0425f,select(0.0475f,0.0350f,nozzle==4u),nozzle==3u);return bottom+(0.0200f-bottom)*z/0.04f;}
  return 0.0200f+(0.0550f-0.0200f)*(z-0.04f)/0.14f;
}
fn nozzleOuterRadius(nozzle:u32,zNoz:f32)->f32 {
  let z=clamp(zNoz,0.0f,0.18f);
  if(z<=0.04f){let bottom=select(0.0465f,select(0.0515f,0.0390f,nozzle==4u),nozzle==3u);return bottom+(0.0240f-bottom)*z/0.04f;}
  return 0.0240f+(0.0590f-0.0240f)*(z-0.04f)/0.14f;
}
`;

const SHADER_FVM_PREDICT = /* wgsl */`${FVM_COMMON}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> s:array<Cell>;
@group(0) @binding(2) var<storage,read_write> us:array<vec4f>;
@group(0) @binding(3) var<storage,read> connection:array<vec4f>;
@group(0) @binding(4) var<storage,read> neighbors:array<i32>;
@group(0) @binding(5) var<storage,read> cellDist:array<vec4f>;
fn neighborIndex(c:u32, slot:u32)->u32 { return u32(neighbors[c*6u+slot]); }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3u) {
  let c=g.x; if(c>=p.N){return;} let q=cellCoord(c,p.NX); let a=s[c]; let link=connection[c];
  if(a.pos.w>0.5f){
    if(link.w>0.5f){
      let nozzleCell=s[u32(link.x)];
      // ★冷風の連続的吹き出しを保証:
      // 以前は室内の圧力上昇によって吹き出しを停止させる非物理的なブレーキ(dynamicFactor)が存在し、
      // 噴流が数ステップで減速・完全停止してしまう原因となっていた。
      // ノズル最下層セルの速度を直接伝達し、連続の式による設計出口速度(V_IN)を確実に維持する。
      var vOut = nozzleCell.vel.xyz;
      let minSpeed = abs(p.V_IN) * 0.75f;
      if (abs(vOut.z) < minSpeed) {
        vOut.z = -abs(p.V_IN);
      }
      us[c]=vec4f(vOut, 0.0f);
    } else {
      us[c]=a.vel;
    }
    return;
  }
  if(c<p.NX*p.NX*p.NX && (q.x==0 || q.x==i32(p.NX)-1 || q.y==0 || q.y==i32(p.NX)-1 || q.z==i32(p.NX)-1)){us[c]=a.vel;return;}
  let h1=cellDist[c*2u]; let h2=cellDist[c*2u+1u];
  let hxE=h1.x; let hxW=h1.y; let hyN=h1.z; let hyS=h1.w;
  let hzU=h2.x; let hzD=h2.y;
  let e=s[neighborIndex(c,0u)].vel.xyz;let w=s[neighborIndex(c,1u)].vel.xyz;
  let no=s[neighborIndex(c,2u)].vel.xyz;let so=s[neighborIndex(c,3u)].vel.xyz;
  let up=s[neighborIndex(c,4u)].vel.xyz;let dn=s[neighborIndex(c,5u)].vel.xyz;let v=a.vel.xyz;
  let dudx=select((e.x-v.x)/hxE,(v.x-w.x)/hxW,v.x>=0.0f);
  let dudy=select((no.x-v.x)/hyN,(v.x-so.x)/hyS,v.y>=0.0f);
  let dudz=select((up.x-v.x)/hzU,(v.x-dn.x)/hzD,v.z>=0.0f);
  let dvdx=select((e.y-v.y)/hxE,(v.y-w.y)/hxW,v.x>=0.0f);
  let dvdy=select((no.y-v.y)/hyN,(v.y-so.y)/hyS,v.y>=0.0f);
  let dvdz=select((up.y-v.y)/hzU,(v.y-dn.y)/hzD,v.z>=0.0f);
  let dwdx=select((e.z-v.z)/hxE,(v.z-w.z)/hxW,v.x>=0.0f);
  let dwdy=select((no.z-v.z)/hyN,(v.z-so.z)/hyS,v.y>=0.0f);
  let dwdz=select((up.z-v.z)/hzU,(v.z-dn.z)/hzD,v.z>=0.0f);
  let lapX=2.0f*((e.x/hxE-v.x*(1.0f/hxE+1.0f/hxW)+w.x/hxW)/(hxE+hxW)+(no.x/hyN-v.x*(1.0f/hyN+1.0f/hyS)+so.x/hyS)/(hyN+hyS)+(up.x/hzU-v.x*(1.0f/hzU+1.0f/hzD)+dn.x/hzD)/(hzU+hzD));
  let lapY=2.0f*((e.y/hxE-v.y*(1.0f/hxE+1.0f/hxW)+w.y/hxW)/(hxE+hxW)+(no.y/hyN-v.y*(1.0f/hyN+1.0f/hyS)+so.y/hyS)/(hyN+hyS)+(up.y/hzU-v.y*(1.0f/hzU+1.0f/hzD)+dn.y/hzD)/(hzU+hzD));
  let lapZ=2.0f*((e.z/hxE-v.z*(1.0f/hxE+1.0f/hxW)+w.z/hxW)/(hxE+hxW)+(no.z/hyN-v.z*(1.0f/hyN+1.0f/hyS)+so.z/hyS)/(hyN+hyS)+(up.z/hzU-v.z*(1.0f/hzU+1.0f/hzD)+dn.z/hzD)/(hzU+hzD));
  let lap=vec3f(lapX,lapY,lapZ);
  let buoy=p.BETA*(a.aux.x-p.T_REF)*p.G;

  // 局所CFL条件による時間刻み制限 (特にノズル微細セルでの爆発を防止)
  let minH=min(min(min(hxE,hxW),min(hyN,hyS)),min(hzU,hzD));
  let vMag=max(length(v),0.05f);
  let dt_eff=min(p.DT, 0.40f*minH/vMag);

  var out=v+dt_eff*(-vec3f(dot(v,vec3f(dudx,dudy,dudz)),dot(v,vec3f(dvdx,dvdy,dvdz)),dot(v,vec3f(dwdx,dwdy,dwdz)))+p.NU*lap+vec3f(0.0f,0.0f,-p.G+buoy));

  // ノズル細密セルでは、準一次元流の流線に沿う接線方向へ動径速度を補正する。
  // ★重要: 中心(r=0)では対称性から vr=0 でなければならず、壁面(r=rIn)で vr=drdz*vz となる。
  // 以前は全セルで一律に vr=drdz*vz としていたため中心軸で左右に引き裂かれ、
  // スロート部でのdrdz符号反転に伴い層ごとの激しい縞模様(発散)を引き起こしていた。
  if(c>=p.NX*p.NX*p.NX && a.pos.w<0.5f){
    let zn=clamp(a.pos.z-p.DOM,0.0f,0.18f);
    let dz=0.002f;
    let r0=nozzleInnerRadius(p.NOZZLE,max(0.0f,zn-dz));
    let r1=nozzleInnerRadius(p.NOZZLE,min(0.18f,zn+dz));
    let rIn=nozzleInnerRadius(p.NOZZLE,zn);
    let drdz=(r1-r0)/(2.0f*dz);
    let radial=vec2f(a.pos.x-p.IN_CX,a.pos.y-p.IN_CY);
    let rr=length(radial);
    if(rr>1e-5f && rIn>1e-4f){
      let er=radial/rr;
      let xi=clamp(rr/rIn, 0.0f, 1.0f);
      let targetVr=xi*drdz*out.z;
      let currentRadial=dot(out.xy,er);
      let correctedRadial=out.xy+er*((targetVr-currentRadial)*0.4f);
      out=vec3f(correctedRadial.x,correctedRadial.y,out.z);
    }
  }

  // 高周波チェッカーボード振動(市松模様)の散逸平滑化フィルタ
  var vAvg = vec3f(0.0f); var nFluid = 0.0f;
  let n0 = s[neighborIndex(c,0u)]; if(n0.pos.w<0.5f){ vAvg += n0.vel.xyz; nFluid += 1.0f; }
  let n1 = s[neighborIndex(c,1u)]; if(n1.pos.w<0.5f){ vAvg += n1.vel.xyz; nFluid += 1.0f; }
  let n2 = s[neighborIndex(c,2u)]; if(n2.pos.w<0.5f){ vAvg += n2.vel.xyz; nFluid += 1.0f; }
  let n3 = s[neighborIndex(c,3u)]; if(n3.pos.w<0.5f){ vAvg += n3.vel.xyz; nFluid += 1.0f; }
  let n4 = s[neighborIndex(c,4u)]; if(n4.pos.w<0.5f){ vAvg += n4.vel.xyz; nFluid += 1.0f; }
  let n5 = s[neighborIndex(c,5u)]; if(n5.pos.w<0.5f){ vAvg += n5.vel.xyz; nFluid += 1.0f; }
  if(nFluid > 0.0f){
    out = mix(out, vAvg / nFluid, 0.035f);
  }

  let vl=length(out);if(vl>35.0f){out*=35.0f/vl;} us[c]=vec4f(out,0.0f);
}
`;

const SHADER_FVM_DIVERGENCE = /* wgsl */`${FVM_COMMON}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> s:array<Cell>;
@group(0) @binding(2) var<storage,read> us:array<vec4f>;
@group(0) @binding(3) var<storage,read_write> div:array<f32>;
@group(0) @binding(4) var<storage,read> neighbors:array<i32>;
@group(0) @binding(5) var<storage,read> cellDist:array<vec4f>;
fn neighborIndex(c:u32, slot:u32)->u32 { return u32(neighbors[c*6u+slot]); }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3u){
  let c=g.x;if(c>=p.N){return;}let q=cellCoord(c,p.NX);
  let roomN=p.NX*p.NX*p.NX;
  let uIdx=neighborIndex(c,4u);
  let isCeilInlet=(c<roomN)&&(q.z==i32(p.NX)-1)&&(uIdx>=roomN);

  if((s[c].pos.w>0.5f&&!isCeilInlet)||(c<roomN&&!isCeilInlet&&(q.x==0||q.x==i32(p.NX)-1||q.y==0||q.y==i32(p.NX)-1||q.z==0||q.z==i32(p.NX)-1))){div[c]=0.0f;return;}
  let h1=cellDist[c*2u]; let h2=cellDist[c*2u+1u];
  let hxE=h1.x; let hxW=h1.y; let hyN=h1.z; let hyS=h1.w;
  let hzU=h2.x; let hzD=h2.y;

  let uC=us[c].xyz;
  let sE=s[neighborIndex(c,0u)]; let uE=select(us[neighborIndex(c,0u)].xyz, vec3f(0.0f, uC.y, uC.z), sE.pos.w>0.5f&&sE.pos.w<1.5f);
  let sW=s[neighborIndex(c,1u)]; let uW=select(us[neighborIndex(c,1u)].xyz, vec3f(0.0f, uC.y, uC.z), sW.pos.w>0.5f&&sW.pos.w<1.5f);
  let sN=s[neighborIndex(c,2u)]; let uN=select(us[neighborIndex(c,2u)].xyz, vec3f(uC.x, 0.0f, uC.z), sN.pos.w>0.5f&&sN.pos.w<1.5f);
  let sS=s[neighborIndex(c,3u)]; let uS=select(us[neighborIndex(c,3u)].xyz, vec3f(uC.x, 0.0f, uC.z), sS.pos.w>0.5f&&sS.pos.w<1.5f);
  let sU=s[neighborIndex(c,4u)]; let uU=us[neighborIndex(c,4u)].xyz;
  let sD=s[neighborIndex(c,5u)]; let uD=us[neighborIndex(c,5u)].xyz;

  let dudx=(uE.x-uW.x)/(hxE+hxW);
  let dvdy=(uN.y-uS.y)/(hyN+hyS);
  let dwdz=(uU.z-uD.z)/(hzU+hzD);
  div[c]=clamp((dudx+dvdy+dwdz)*p.RHO0/p.DT, -40000.0f, 40000.0f);
}
`;

const SHADER_FVM_PRESSURE = /* wgsl */`${FVM_COMMON}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> s:array<Cell>;
@group(0) @binding(2) var<storage,read> div:array<f32>;
@group(0) @binding(3) var<storage,read> pin:array<f32>;
@group(0) @binding(4) var<storage,read_write> pout:array<f32>;
@group(0) @binding(5) var<storage,read> neighbors:array<i32>;
@group(0) @binding(6) var<storage,read> cellDist:array<vec4f>;
fn neighborIndex(c:u32, slot:u32)->u32 { return u32(neighbors[c*6u+slot]); }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3u){
  let c=g.x;if(c>=p.N){return;}let q=cellCoord(c,p.NX);let a=s[c];
  let roomN=p.NX*p.NX*p.NX;
  let uIdx=neighborIndex(c,4u);
  let isCeilInlet=(c<roomN)&&(q.z==i32(p.NX)-1)&&(uIdx>=roomN);

  // ★天井開口セル: ノズル最下層(出口)の圧力をそのまま受け取り、室内側へ100%連続伝達する
  if(isCeilInlet){
    pout[c]=pin[uIdx];
    return;
  }
  // 天井開口部以外の壁セル、および室内境界壁面はDirichlet条件(p=0)
  if(a.pos.w>0.5f||(c<roomN&&(q.x==0||q.x==i32(p.NX)-1||q.y==0||q.y==i32(p.NX)-1||q.z==0||q.z==i32(p.NX)-1))){pout[c]=0.0f;return;}
  let h1=cellDist[c*2u]; let h2=cellDist[c*2u+1u];
  let hxE=h1.x; let hxW=h1.y; let hyN=h1.z; let hyS=h1.w;
  let hzU=h2.x; let hzD=h2.y;
  let aE=2.0f/(hxE*(hxE+hxW)); let aW=2.0f/(hxW*(hxE+hxW));
  let aN=2.0f/(hyN*(hyN+hyS)); let aS=2.0f/(hyS*(hyN+hyS));
  let aU=2.0f/(hzU*(hzU+hzD)); let aD=2.0f/(hzD*(hzU+hzD));
  let aC=aE+aW+aN+aS+aU+aD;

  // 壁境界条件: 壁面では dp/dn = 0 (Neumann条件)
  // 隣接セルが壁の場合、壁セルの圧力を中心セルの圧力 pC と等しいとみなす
  let pC=pin[c];
  let sE=s[neighborIndex(c,0u)]; let pE=select(pin[neighborIndex(c,0u)], pC, sE.pos.w>0.5f&&sE.pos.w<1.5f);
  let sW=s[neighborIndex(c,1u)]; let pW=select(pin[neighborIndex(c,1u)], pC, sW.pos.w>0.5f&&sW.pos.w<1.5f);
  let sN=s[neighborIndex(c,2u)]; let pN=select(pin[neighborIndex(c,2u)], pC, sN.pos.w>0.5f&&sN.pos.w<1.5f);
  let sS=s[neighborIndex(c,3u)]; let pS=select(pin[neighborIndex(c,3u)], pC, sS.pos.w>0.5f&&sS.pos.w<1.5f);
  let sU=s[neighborIndex(c,4u)]; let pU=select(pin[neighborIndex(c,4u)], pC, sU.pos.w>0.5f&&sU.pos.w<1.5f);
  let sD=s[neighborIndex(c,5u)]; let pD=select(pin[neighborIndex(c,5u)], pC, sD.pos.w>0.5f&&sD.pos.w<1.5f);

  let newP = (aE*pE+aW*pW+aN*pN+aS*pS+aU*pU+aD*pD-div[c])/aC;
  pout[c]=clamp(newP, -800.0f, 800.0f);
}
`;

const SHADER_FVM_CORRECT = /* wgsl */`${FVM_COMMON}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> s:array<Cell>;
@group(0) @binding(2) var<storage,read> us:array<vec4f>;
@group(0) @binding(3) var<storage,read> pr:array<f32>;
@group(0) @binding(4) var<storage,read_write> out:array<Cell>;
@group(0) @binding(5) var<storage,read> neighbors:array<i32>;
@group(0) @binding(6) var<storage,read> cellDist:array<vec4f>;
fn neighborIndex(c:u32, slot:u32)->u32 { return u32(neighbors[c*6u+slot]); }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3u){
  let c=g.x;if(c>=p.N){return;}let a=s[c];
  let roomN=p.NX*p.NX*p.NX;
  let uIdx=neighborIndex(c,4u);
  let isCeilInlet=(c<roomN)&&(uIdx>=roomN);

  if(a.pos.w>0.5f){
    if(isCeilInlet){
      let nozPr = pr[uIdx];
      var inlet=a;
      inlet.vel=vec4f(us[c].xyz, nozPr); // ノズル出口圧力を確実に保持
      inlet.aux.x=p.T_IN;
      inlet.aux.z=length(inlet.vel.xyz);
      out[c]=inlet;
    } else {
      var fixedCell=a;
      if(a.pos.w>1.5f){ fixedCell.aux.x=p.T_IN; }
      out[c]=fixedCell;
    }
    return;
  }
  let h1=cellDist[c*2u]; let h2=cellDist[c*2u+1u];
  let hxE=h1.x; let hxW=h1.y; let hyN=h1.z; let hyS=h1.w;
  let hzU=h2.x; let hzD=h2.y;

  // 壁境界条件: 壁面では dp/dn = 0 (Neumann条件)
  let pC=pr[c];
  let sE=s[neighborIndex(c,0u)]; let pE=select(pr[neighborIndex(c,0u)], pC, sE.pos.w>0.5f&&sE.pos.w<1.5f);
  let sW=s[neighborIndex(c,1u)]; let pW=select(pr[neighborIndex(c,1u)], pC, sW.pos.w>0.5f&&sW.pos.w<1.5f);
  let sN=s[neighborIndex(c,2u)]; let pN=select(pr[neighborIndex(c,2u)], pC, sN.pos.w>0.5f&&sN.pos.w<1.5f);
  let sS=s[neighborIndex(c,3u)]; let pS=select(pr[neighborIndex(c,3u)], pC, sS.pos.w>0.5f&&sS.pos.w<1.5f);
  let sU=s[neighborIndex(c,4u)]; let pU=select(pr[neighborIndex(c,4u)], pC, sU.pos.w>0.5f&&sU.pos.w<1.5f);
  let sD=s[neighborIndex(c,5u)]; let pD=select(pr[neighborIndex(c,5u)], pC, sD.pos.w>0.5f&&sD.pos.w<1.5f);

  let gradX=(pE-pW)/(hxE+hxW);
  let gradY=(pN-pS)/(hyN+hyS);
  let gradZ=(pU-pD)/(hzU+hzD);
  let v_corr = us[c].xyz - p.DT/p.RHO0*vec3f(gradX,gradY,gradZ);
  let v_len = length(v_corr);
  let maxV = 35.0f;
  let v_safe = select(v_corr, v_corr * (maxV / v_len), v_len > maxV);
  var b=a; b.vel=vec4f(v_safe,pr[c]); b.aux.z=length(v_safe); out[c]=b;
}
`;

const SHADER_FVM_TEMPERATURE = /* wgsl */`${FVM_COMMON}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> s:array<Cell>;
@group(0) @binding(2) var<storage,read_write> out:array<Cell>;
@group(0) @binding(3) var<storage,read> neighbors:array<i32>;
@group(0) @binding(4) var<storage,read> cellDist:array<vec4f>;
fn neighborIndex(c:u32, slot:u32)->u32 { return u32(neighbors[c*6u+slot]); }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3u){
  let c=g.x;if(c>=p.N){return;}let a=s[c];
  if(a.pos.w>0.5f){
    var fixedCell=a;
    if(a.pos.w>1.5f){ fixedCell.aux.x=p.T_IN; }
    out[c]=fixedCell;
    return;
  }
  let q=cellCoord(c,p.NX);
  if(c<p.NX*p.NX*p.NX&&(q.x==0||q.x==i32(p.NX)-1||q.y==0||q.y==i32(p.NX)-1||q.z==i32(p.NX)-1)){
    var boundaryCell=a;
    if(a.pos.w>1.5f){ boundaryCell.aux.x=p.T_IN; }
    out[c]=boundaryCell;
    return;
  }
  let h1=cellDist[c*2u]; let h2=cellDist[c*2u+1u];
  let hxE=h1.x; let hxW=h1.y; let hyN=h1.z; let hyS=h1.w;
  let hzU=h2.x; let hzD=h2.y;
  let t=a.aux.x; let v=a.vel.xyz;
  let sE=s[neighborIndex(c,0u)]; let e=select(sE.aux.x, t, sE.pos.w>0.5f&&sE.pos.w<1.5f);
  let sW=s[neighborIndex(c,1u)]; let w=select(sW.aux.x, t, sW.pos.w>0.5f&&sW.pos.w<1.5f);
  let sN=s[neighborIndex(c,2u)]; let no=select(sN.aux.x, t, sN.pos.w>0.5f&&sN.pos.w<1.5f);
  let sS=s[neighborIndex(c,3u)]; let so=select(sS.aux.x, t, sS.pos.w>0.5f&&sS.pos.w<1.5f);
  let sU=s[neighborIndex(c,4u)]; let up=select(sU.aux.x, t, sU.pos.w>0.5f&&sU.pos.w<1.5f);
  let sD=s[neighborIndex(c,5u)]; let dn=select(sD.aux.x, t, sD.pos.w>0.5f&&sD.pos.w<1.5f);
  let gx=select((e-t)/hxE,(t-w)/hxW,v.x>=0.0f);
  let gy=select((no-t)/hyN,(t-so)/hyS,v.y>=0.0f);
  let gz=select((up-t)/hzU,(t-dn)/hzD,v.z>=0.0f);
  let lapT=2.0f*((e/hxE-t*(1.0f/hxE+1.0f/hxW)+w/hxW)/(hxE+hxW)+(no/hyN-t*(1.0f/hyN+1.0f/hyS)+so/hyS)/(hyN+hyS)+(up/hzU-t*(1.0f/hzU+1.0f/hzD)+dn/hzD)/(hzU+hzD));
  let advT=dot(v,vec3f(gx,gy,gz));
  let minH=min(min(min(hxE,hxW),min(hyN,hyS)),min(hzU,hzD));
  let vMag=max(length(v),0.01f);
  let dt_local=min(p.DT, 0.45f*minH/vMag);
  var b=a; b.aux.x=clamp(t+dt_local*(-advT+p.ALPHA*lapT), min(p.T_IN,p.T_REF), max(p.T_IN,p.T_REF));
  b.aux.z=length(b.vel.xyz);out[c]=b;
}
`;

const SHADER_FVM_SLICE = /* wgsl */`${FVM_COMMON}
struct Camera { mvp:mat4x4f, viewPos:vec4f, slice_mode:f32, slice_pos:f32, slice_thick:f32, view_mode:f32, };
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> cells:array<Cell>;
@group(0) @binding(2) var<uniform> cam:Camera;
@group(0) @binding(3) var<storage,read_write> values:array<f32>;
@group(0) @binding(4) var<storage,read> gridAxes:array<f32>;

fn findAxisIndex(val:f32, offset:u32, n:u32) -> u32 {
  var low:u32 = 0u;
  var high:u32 = n - 1u;
  while(low < high){
    let mid:u32 = (low + high + 1u) / 2u;
    if(gridAxes[offset + mid] <= val){
      low = mid;
    }else{
      high = mid - 1u;
    }
  }
  return min(low, n - 2u);
}

@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) g:vec3u){
  if(g.x>=65u||g.y>=65u){return;} let id=g.y*65u+g.x;
  let u=f32(g.x)/64.0f*p.DOM; let z=f32(g.y)/64.0f*p.DOM_Z;
  let sample=select(vec3f(cam.slice_pos,u,z),vec3f(u,cam.slice_pos,z),cam.slice_mode>1.5f);
  if(z>p.DOM){
    var nearest=p.NX*p.NX*p.NX;var best=1.0f;
    for(var ni=p.NX*p.NX*p.NX;ni<p.N;ni++){
      let d=cells[ni].pos.xyz-sample;let d2=dot(d,d);
      if(d2<best){best=d2;nearest=ni;}
    }
    let nc=cells[nearest];
    if(best>0.005f*0.005f*2.0f||nc.pos.w>0.5f){values[id]=-1.0f;return;}
    var nv=nc.aux.x;
    if(p.DISP_MODE==1u){nv=nc.aux.z;}else if(p.DISP_MODE==2u){nv=abs(nc.vel.w);}
    if(p.DISP_MODE==0u){nv=(nv-p.T_MIN)/(p.T_MAX-p.T_MIN);}else if(p.DISP_MODE==1u){nv/=p.V_MAX;}else{nv/=500.0f;}
    values[id]=clamp(nv,0.0f,1.0f);
    return;
  }
  let axCount = p.NX + 1u;
  let ix = findAxisIndex(sample.x, 0u, axCount);
  let iy = findAxisIndex(sample.y, axCount, axCount);
  let iz = findAxisIndex(sample.z, axCount * 2u, axCount);
  let x0 = gridAxes[ix]; let x1 = gridAxes[ix + 1u];
  let y0 = gridAxes[axCount + iy]; let y1 = gridAxes[axCount + iy + 1u];
  let z0 = gridAxes[axCount * 2u + iz]; let z1 = gridAxes[axCount * 2u + iz + 1u];
  let fx = clamp((sample.x - x0) / max(x1 - x0, 1e-6f), 0.0f, 1.0f);
  let fy = clamp((sample.y - y0) / max(y1 - y0, 1e-6f), 0.0f, 1.0f);
  let fz = clamp((sample.z - z0) / max(z1 - z0, 1e-6f), 0.0f, 1.0f);
  let frac = vec3f(fx, fy, fz);
  let base = vec3i(i32(ix), i32(iy), i32(iz));
  var sum=0.0f;var weightSum=0.0f;
  for(var dx:i32=0;dx<=1;dx++){
    for(var dy:i32=0;dy<=1;dy++){
      for(var dz:i32=0;dz<=1;dz++){
        let wx=select(1.0f-frac.x,frac.x,dx==1);
        let wy=select(1.0f-frac.y,frac.y,dy==1);
        let wz=select(1.0f-frac.z,frac.z,dz==1);
        let weight=wx*wy*wz;
        let c=cells[index3(base.x+dx,base.y+dy,base.z+dz,p.NX)];
        if(abs(c.pos.w-1.0f)>0.5f){
          var cv=c.aux.x;
          if(c.pos.w>1.5f&&p.DISP_MODE==0u){cv=p.T_IN;}
          else if(c.pos.w>1.5f&&p.DISP_MODE==1u){cv=p.V_IN;}
          else if(c.pos.w>1.5f&&p.DISP_MODE==2u){cv=0.0f;}
          else if(p.DISP_MODE==1u){cv=c.aux.z;}
          else if(p.DISP_MODE==2u){cv=abs(c.vel.w);}
          sum+=cv*weight;weightSum+=weight;
        }
      }
    }
  }
  var v=sum/max(weightSum,0.0001f);
  if(p.DISP_MODE==1u){
    let radial=select(abs(sample.y-p.IN_CY),abs(sample.x-p.IN_CX),cam.slice_mode>1.5f);
    let blend=clamp((z-(p.DOM-0.1f))/0.1f,0.0f,1.0f);
    if(radial<=p.IN_R){v=mix(v,p.V_IN,blend);}
  }
  if(p.DISP_MODE==0u){v=(v-p.T_MIN)/(p.T_MAX-p.T_MIN);}else if(p.DISP_MODE==1u){v/=p.V_MAX;}else{v/=500.0f;}
  values[id]=clamp(v,0.0f,1.0f);
}
`;

// ─────────────────────────────────────────────────────────────────
// 断面セル描画 (visualMode===1 用)
//   従来の断面表示は粒子状ビルボード(円)を並べていたため、非均一格子の
//   粗い領域でセル間に隙間が生じていた。本シェーダーは gridAxes (実際の
//   格子面座標) からセル矩形そのものを三角形2枚で描くため、断面全体が
//   隙間なく埋まる。フラグメント側でセル境界を細線として描き、
//   「格子フレームワーク」として読めるようにしている。
//
//   インスタンス: NX×NX (面内水平方向 a × 鉛直方向 b)
//   頂点        : 6 (三角形2枚)
// ─────────────────────────────────────────────────────────────────
const SHADER_FVM_SLICE_CELL = /* wgsl */`${FVM_COMMON}
struct Camera { mvp:mat4x4f, viewPos:vec4f, slice_mode:f32, slice_pos:f32, slice_thick:f32, view_mode:f32, };
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> cells:array<Cell>;
@group(0) @binding(2) var<uniform> cam:Camera;
@group(0) @binding(3) var<storage,read> gridAxes:array<f32>;

fn jetC(t:f32)->vec3f{
  let x=clamp(t,0.0f,1.0f);
  var r=0.0f; var g=0.0f; var b=0.0f;
  if(x<0.125f){ b=0.5f+4.0f*x; }
  else if(x<0.375f){ g=4.0f*(x-0.125f); b=1.0f; }
  else if(x<0.625f){ r=4.0f*(x-0.375f); g=1.0f; b=1.0f-4.0f*(x-0.375f); }
  else if(x<0.875f){ r=1.0f; g=1.0f-4.0f*(x-0.625f); }
  else { r=1.0f-4.0f*(x-0.875f); }
  return vec3f(r,g,b);
}

// 単調増加の格子面座標配列から val を含むセル添字を二分探索
fn axisIndex(val:f32, offset:u32, n:u32)->u32{
  var low:u32=0u; var high:u32=n-1u;
  while(low<high){
    let mid:u32=(low+high+1u)/2u;
    if(gridAxes[offset+mid]<=val){ low=mid; } else { high=mid-1u; }
  }
  return min(low,n-2u);
}

struct SliceCellOut {
  @builtin(position) pos:vec4f,
  @location(0) color:vec3f,
  @location(1) uv:vec2f,
};

@vertex fn vs_main(@builtin(vertex_index) vid:u32, @builtin(instance_index) iid:u32)->SliceCellOut{
  let NX=p.NX; let ax=NX+1u;
  let a=iid%NX;          // 面内 水平方向のセル添字
  let b=iid/NX;          // 鉛直(z)方向のセル添字
  let yz = cam.slice_mode < 1.5f;      // true: YZ断面(x固定) / false: XZ断面(y固定)
  let hOff = select(0u, ax, yz);       // 面内水平軸: YZ→y軸, XZ→x軸
  let fOff = select(ax, 0u, yz);       // 固定軸  : YZ→x軸, XZ→y軸
  let fIdx = axisIndex(cam.slice_pos, fOff, ax);

  let corner=array<vec2f,6>(
    vec2f(0.0f,0.0f),vec2f(1.0f,0.0f),vec2f(0.0f,1.0f),
    vec2f(1.0f,0.0f),vec2f(1.0f,1.0f),vec2f(0.0f,1.0f))[vid];

  // セル矩形の4隅を実際の格子面座標から取る → 隣接セルと端点が共有され隙間ゼロ
  let h0=gridAxes[hOff+a];      let h1=gridAxes[hOff+a+1u];
  let z0=gridAxes[ax*2u+b];     let z1=gridAxes[ax*2u+b+1u];
  let hh=mix(h0,h1,corner.x);   let zz=mix(z0,z1,corner.y);

  var i:u32; var j:u32;
  if(yz){ i=fIdx; j=a; } else { i=a; j=fIdx; }
  let c=cells[index3(i32(i),i32(j),i32(b),NX)];

  // 物理座標 → THREE系 (x-1, z-1, y-1)
  var world:vec3f;
  if(yz){ world=vec3f(cam.slice_pos-1.0f, zz-1.0f, hh-1.0f); }
  else  { world=vec3f(hh-1.0f, zz-1.0f, cam.slice_pos-1.0f); }

  let ptype=c.pos.w;
  var v=c.aux.x;
  if(p.DISP_MODE==1u){ v=c.aux.z; } else if(p.DISP_MODE==2u){ v=abs(c.vel.w); }
  if(ptype>1.5f){
    // 流入セルの境界値: 温度はT_IN、速度はV_IN、圧力はセル自身の圧力(abs(c.vel.w))をそのまま表示
    if(p.DISP_MODE==0u){ v=p.T_IN; } else if(p.DISP_MODE==1u){ v=p.V_IN; }
  }
  var t:f32;
  if(p.DISP_MODE==0u){ t=(v-p.T_MIN)/max(p.T_MAX-p.T_MIN,1e-5f); }
  else if(p.DISP_MODE==1u){ t=v/max(p.V_MAX,1e-5f); }
  else {
    let pScale = max(0.5f * p.RHO0 * p.V_MAX * p.V_MAX, 40.0f);
    t = v / pScale;
  }

  var col=jetC(t);
  // 壁セルも塗りつぶす(抜かない)ことで断面に穴が開かないようにする
  if(ptype>0.5f&&ptype<1.5f){ col=vec3f(0.20f,0.23f,0.30f); }

  var out:SliceCellOut;
  out.pos=cam.mvp*vec4f(world,1.0f);
  out.color=col;
  out.uv=corner;
  return out;
}

@fragment fn fs_main(@location(0) color:vec3f, @location(1) uv:vec2f)->@location(0) vec4f{
  // セル境界を画面上一定幅の細線として描く (格子フレームワーク表現)
  let d=min(min(uv.x,1.0f-uv.x),min(uv.y,1.0f-uv.y));
  let w=max(fwidth(d),1e-6f);
  let edge=1.0f-smoothstep(0.0f,w*1.5f,d);
  return vec4f(mix(color,vec3f(0.05f,0.07f,0.11f),edge*0.85f),1.0f);
}
`;

// ─────────────────────────────────────────────────────────────────
// ノズル微細セル(統一セル配列の roomN 以降)の断面スカラー描画。
// 室内側 SHADER_FVM_SLICE_CELL と同じ配色規則を、ノズルのボクセル
// 1個ずつに直接適用する。ノズルセルは非構造格子(円形にくり抜かれた
// 立方体格子)なので gridAxes は使わず、セル自身の中心座標 pos.xyz と
// 固定半セル幅 h からクアッドを組む。スライス面に交差しないセルは
// NDC範囲外へ飛ばしてクリップさせ、実質「非表示」にする。
// ─────────────────────────────────────────────────────────────────
const SHADER_FVM_SLICE_CELL_NOZZLE = /* wgsl */`${FVM_COMMON}
struct Camera { mvp:mat4x4f, viewPos:vec4f, slice_mode:f32, slice_pos:f32, slice_thick:f32, view_mode:f32, };
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> cells:array<Cell>;
@group(0) @binding(2) var<uniform> cam:Camera;

fn jetCN(t:f32)->vec3f{
  let x=clamp(t,0.0f,1.0f);
  var r=0.0f; var g=0.0f; var b=0.0f;
  if(x<0.125f){ b=0.5f+4.0f*x; }
  else if(x<0.375f){ g=4.0f*(x-0.125f); b=1.0f; }
  else if(x<0.625f){ r=4.0f*(x-0.375f); g=1.0f; b=1.0f-4.0f*(x-0.375f); }
  else if(x<0.875f){ r=1.0f; g=1.0f-4.0f*(x-0.625f); }
  else { r=1.0f-4.0f*(x-0.875f); }
  return vec3f(r,g,b);
}

struct Out { @builtin(position) pos:vec4f, @location(0) color:vec3f, @location(1) uv:vec2f, };

@vertex fn vs_main(@builtin(vertex_index) vid:u32, @builtin(instance_index) iid:u32)->Out{
  let roomN=p.NX*p.NX*p.NX;
  let c=cells[roomN+iid];
  let h=0.18f/48.0f*0.5f;   // _createNozzleState と同じ .18/48 格子の半セル幅

  let corner=array<vec2f,6>(
    vec2f(0.0f,0.0f),vec2f(1.0f,0.0f),vec2f(0.0f,1.0f),
    vec2f(1.0f,0.0f),vec2f(1.0f,1.0f),vec2f(0.0f,1.0f))[vid];

  let yz = cam.slice_mode < 1.5f;   // true: YZ断面(物理x固定) / false: XZ断面(物理y固定)
  let fixedCoord = select(c.pos.y, c.pos.x, yz);
  var out:Out;

  if(abs(fixedCoord-cam.slice_pos) > h){
    out.pos=vec4f(2.0f,2.0f,2.0f,1.0f); out.color=vec3f(0.0f); out.uv=vec2f(0.0f); return out; // クリップ範囲外へ
  }

  var world:vec3f;
  if(yz){
    world=vec3f(cam.slice_pos-1.0f,
                mix(c.pos.z-1.0f-h,c.pos.z-1.0f+h,corner.y),
                mix(c.pos.y-1.0f-h,c.pos.y-1.0f+h,corner.x));
  } else {
    world=vec3f(mix(c.pos.x-1.0f-h,c.pos.x-1.0f+h,corner.x),
                mix(c.pos.z-1.0f-h,c.pos.z-1.0f+h,corner.y),
                cam.slice_pos-1.0f);
  }

  let ptype=c.pos.w;
  var v=c.aux.x;
  if(p.DISP_MODE==1u){ v=c.aux.z; } else if(p.DISP_MODE==2u){ v=abs(c.vel.w); }
  if(ptype>1.5f){
    // 流入セルの境界値: 温度はT_IN、速度はV_IN、圧力はセル自身の圧力(abs(c.vel.w))をそのまま表示
    if(p.DISP_MODE==0u){ v=p.T_IN; } else if(p.DISP_MODE==1u){ v=p.V_IN; }
  }
  var t:f32;
  if(p.DISP_MODE==0u){ t=(v-p.T_MIN)/max(p.T_MAX-p.T_MIN,1e-5f); }
  else if(p.DISP_MODE==1u){ t=v/max(p.V_MAX,1e-5f); }
  else {
    let pScale = max(0.5f * p.RHO0 * p.V_MAX * p.V_MAX, 40.0f);
    t = v / pScale;
  }

  var col=jetCN(t);
  if(ptype>0.5f&&ptype<1.5f){ col=vec3f(0.20f,0.23f,0.30f); } // 壁は塗って穴を作らない

  out.pos=cam.mvp*vec4f(world,1.0f);
  out.color=col;
  out.uv=corner;
  return out;
}

@fragment fn fs_main(@location(0) color:vec3f, @location(1) uv:vec2f)->@location(0) vec4f{
  let d=min(min(uv.x,1.0f-uv.x),min(uv.y,1.0f-uv.y));
  let w=max(fwidth(d),1e-6f);
  let edge=1.0f-smoothstep(0.0f,w*1.5f,d);
  return vec4f(mix(color,vec3f(0.05f,0.07f,0.11f),edge*0.85f),1.0f);
}
`;
