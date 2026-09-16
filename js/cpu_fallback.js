'use strict';
/**
 * cpu_fallback.js
 * WebGPU非対応ブラウザ向け CPU 有限体積法(FVM)流体シミュレーター
 * Three.js (WebGL) でレンダリング
 *
 * ── 手法概要 ─────────────────────────────────────────────────────
 * 旧版は簡略化MPS(粒子法)だったが、本版は構造格子(Cartesianセル中心
 * 格子)上の有限体積法に置き換えている。
 *
 *   1. 各セルは立方体のコントロールボリューム(体積 h³)。
 *      セル面を通過するフラックス(移流・拡散)の収支から時間発展させる
 *      ——これが「有限体積法」の核。等間隔格子なので係数は差分法と
 *      一致するが、離散化の考え方はフラックス収支(FV)に基づく。
 *   2. 移流項: 風上差分(donor-cell)によるセル面フラックス評価。
 *   3. 拡散項: 標準7点ラプラシアン(中心差分)。
 *   4. 圧力-速度連成: Chorin の分離解法(投影法)。
 *        (a) 圧力項を無視して仮速度 u* を計算
 *        (b) ∇²p = (ρ/Δt)∇·u* をヤコビ法で反復的に解く
 *        (c) u = u* − (Δt/ρ)∇p として発散ゼロの速度場へ補正
 *   5. 浮力: Boussinesq近似 (実際に密度は変えず、運動量式に
 *      β(T−T_ref)g の項を加える)。
 *   6. 境界条件:
 *        - 側壁・天井(ノズル部除く): 固体壁 (no-slip, 等温壁 T_ref)
 *        - 天井ノズル部: nozzle.js の断面プロファイル(テーパー内外径、
 *          図2の中心シリンダー)をセル分類にそのまま反映。
 *          r > 外径 → 固体(天井/ノズル胴外側)、
 *          内径 ≦ r ≦ 外径 → 固体(ノズル壁)、
 *          r < 内径(中心シリンダー域は除く) → 流入 Dirichlet
 *          (速度 -Vin, 温度 T_in)。
 *          物理寸法(Ø70〜95mm)は格子セル幅(≈111mm)より小さいため、
 *          旧版は断面全体を強制的にスケールアップして最低限の吹出し
 *          セル数を確保していたが、これはノズル形状ごとにスケール率が
 *          異なり形状比較を歪めてしまう。本版ではノズル内部〜近傍の
 *          セルのみ NOZZLE_SUBSAMPLE^2×3層 のサブグリッドで実寸
 *          (nozzleScale=1)のまま開口率(inletFrac)をサンプリングし、
 *          その開口率で吹出し速度を按分することで、格子解像度を
 *          上げずにノズル内部〜近傍の実効解像度だけを引き上げている
 *          (ボリュームフラクション/多孔質境界的な埋め込みサブグリッド)。
 *          ノズル形状の可視化(_updateNozzleMesh)もソリッドではなく
 *          ワイヤーフレームで描画し、形状の分割線そのものを比較できる
 *          ようにしている。
 *        - 床面: no-slip等温壁(側壁・天井と同様、速度ゼロで固定、
 *          温度T_ref)。旧版は排気口として開放境界(p=0, ゼロ勾配流出)
 *          だったが、床をno-slip固体壁に変更。
 *
 * 可視化層(Three.js の点群・断面サーフェス等)はセル中心を「粒子」と
 * 同じ配列レイアウト(px,py,pz,vx,vy,vz,temp,press,ptype)で保持する
 * ことで、main.js 側や描画コードを変更せずに流用できるようにしている。
 */

class CPUFallbackMPS {
  constructor() {
    // ── 物理定数 ──────────────────────────────────────────────────
    this.DOM    = 2.0;              // 部屋一辺 [m] (室内は厳密に 2.0×2.0×2.0 の立方体)
    this.NOZZLE_H = 0.18;           // ノズル高さ [m] (室内の「上」に積む。室内を侵食しない)
    this.DOM_Z  = 2.18;             // z方向総高さ [m] = 室内 2.0 + ノズル 0.18
    this.NZ_NOZ = 12;               // ノズル区間(z: 2.0→2.18)の格子層数
    this.NX     = 64;               // 格子解像度 (NX×NX×NX セル)。
                                     // 旧NX=46 (97,336セル) → NX=64 (262,144セル)。
                                     // tanh stretching でノズル軸付近の最小格子幅が
                                     // 約 h_min ≈ DOM/NX * 0.08 ≈ 2.5mm 相当になり、
                                     // 均一格子に換算すると ~800セル/方向相当の解像度で
                                     // ノズル出口Ø85mmを直径あたり ~34点で解像できる。
    this.H      = this.DOM / this.NX; // 均一格子の参照幅 [m] (境界判定・DT算出基準)
                                     // 実際の格子幅は xg/yg/zg 配列を使用。
    // 時間刻み: tanh stretching で最小格子幅 h_min が約 DOM/NX * 0.08 になる。
    // 拡散安定条件 DT < h_min²/(6ν): h_min≈0.0025m, ν=0.06 → 約1.7e-5s 以下。
    // 移流 CFL 条件 DT < h_min/|v|max: h_min/20 ≈ 1.25e-4s。
    // → 安全率3を見込んで DT=0.0005s (均一NX=46の0.0007sより少し縮小)。
    this.DT     = 0.0001;           // 時間刻み [s] (拡散・移流安定限界 DT < h_min²/(6ν) を充足)
    this.NU     = 0.005;            // 乱流有効粘度 [m²/s]
    this.ALPHA  = 0.005;            // 乱流有効熱拡散率 [m²/s]
    this.G      = 9.81;
    this.BETA   = 1 / 307;          // 体膨張係数 (Boussinesq近似)
    this.T_REF  = 34.0;             // 室内初期温度・壁温 [°C]
    this.T_IN   = 23.5;             // 吹出し温度 [°C]
    this.RHO0   = 1.2;              // 空気密度 [kg/m³]
    this.PRESSURE_ITERS = 32;       // 圧力ポアソン方程式のヤコビ反復回数/ステップ
    this.NOZZLE_SUBSAMPLE = 12;     // ノズル内部〜近傍セルの開口率算出用サブサンプル数(1辺あたり、×3高さ層)

    // ── セルタイプ ────────────────────────────────────────────────
    // VOID: ノズル区間(z>DOM)のうちノズル胴体より外側にある「存在しない」セル。
    //       ソルバーでは壁と同じ扱い(固定値)、描画では完全に非表示にする。
    //       これにより室内立方体の上面より上に余計なセルが描かれなくなる。
    this.FLUID = 0; this.WALL = 1; this.INLET = 2; this.VOID = 3;

    // z方向セル数 (= NX + NZ_NOZ)。x,y は NX のまま。
    this.NZ = 0;

    // ── フィールド配列 (Float32Array, セル中心) ──────────────────────
    this.N = 0;
    this.px = null; this.py = null; this.pz = null;   // セル中心座標(静的)
    this.vx = null; this.vy = null; this.vz = null;   // 速度 (n)
    this.vxs= null; this.vys= null; this.vzs= null;   // 仮速度 u* (投影法)
    this.press= null; this.temp = null;
    this.ptype= null; // Uint8Array
    this.inletFrac = null; // Float32Array: INLETセルの開口率(サブグリッド平均、0〜1)

    // ── 非均一格子座標配列 (tanh stretching) ──────────────────────
    // 各軸の格子面座標 [m]。セル中心は (xg[i]+xg[i+1])/2。
    // 演算子内の局所格子幅: hE=xg[i+1]-xg[i], hW=xg[i]-xg[i-1] 等。
    this.xg = null;  // Float64Array[NX+1]
    this.yg = null;  // Float64Array[NX+1]
    this.zg = null;  // Float64Array[NX+1]

    // ── ソルバー用スクラッチバッファ ───────────────────────────────
    this._divBuf = null;
    this._pBuf   = null;
    this._tBuf   = null;


    // ── Three.js ──────────────────────────────────────────────────
    this.scene    = null;
    this.camera   = null;
    this.renderer = null;
    this.controls = null;
    this.points   = null; // セルワイヤーフレーム(LineSegments、旧: 点群)
    this.geometry = null;
    this._cubeOffsets = null; // 立方体エッジの単位オフセット(24頂点×xyz、半径1)
    this.nozzleMesh = null;
    this.sectionSurface = null;
    this.sectionSurfaceGeometry = null;
    this.sectionGridLines = null;      // 断面セル境界線(格子フレームワーク)
    this.sectionGridGeometry = null;
    this.sectionCellCount = 0;
    this.vectorLines = null;
    this.vectorGeometry = null;
    this.maxVectors = 6000;

    // ── 状態 ──────────────────────────────────────────────────────
    this.nozzleType  = 1;
    this.displayMode = 1;
    this.visualMode  = 1; // 0=3D粒子, 1=断面, 2=コンター
    this.sliceMode   = 1; // 1=YZ面, 2=XZ面
    this.slicePosition = 1.0;
    this.nozzleFocus = true; // ノズル部拡大トグル(断面・コンター表示時のみ有効)
    this.inletPressure = 60.0; // φ110流入口の加圧圧力 [Pa]
    this.viewSignature = '';
    this.paused      = true;
    this.stepCount   = 0;
    this.substepsPerFrame = 3; // NX=46(97,336セル)化に伴い、旧NX=18比でセル数約17倍・
                                // DT縮小により同一物理時間到達に必要なステップ数も約7倍必要。
                                // ここを増やすとフレームあたりの計算負荷がさらに増すため、
                                // 実時間フレームレートは大幅に低下する前提で据え置いている
                                // (文献は定常解析であり、収束までの体感速度より最終分布の
                                // 一致を優先する方針)。
    this.stats = { tMin: 23.5, tMax: 34, vMax: 0, step: 0, time: 0 };

    // 乱流モデル (k-ε) 定数とフラグ
    this.useTurbulence = false;
    this.C_mu = 0.09;
    this.C_eps1 = 1.44;
    this.C_eps2 = 1.92;
    this.sigma_k = 1.0;
    this.sigma_eps = 1.3;

    // ── エンジン識別 (sim_mode.js の手動切替で参照) ────────────────────
    this.engineMode  = 'cpu';
    this.engineLabel = 'CPU (WebGL / FVM)';
  }

  // WebGL さえあれば常に利用可能
  static isAvailable() {
    try {
      const c = document.createElement('canvas');
      return !!(c.getContext('webgl2') || c.getContext('webgl'));
    } catch (e) { return false; }
  }

  // ────────────────────────────────────────────────────────────────
  //  初期化
  // ────────────────────────────────────────────────────────────────
  init(canvas, nozzleType = 1) {
    this.nozzleType = nozzleType;
    this._initParticles(nozzleType);
    this._initThree(canvas);
    this._initOrbitControls();
    this.updateView();
    return this;
  }

  // ────────────────────────────────────────────────────────────────
  //  格子初期化 (旧: 粒子初期化。メソッド名は互換性のため維持)
  // ────────────────────────────────────────────────────────────────
  _initParticles(nozzleType) {
    const NX = this.NX;
    const nozzle = NOZZLES[nozzleType];
    const core   = nozzle.innerCylinder || null; // 図2のみ存在
    const cx = this.DOM / 2, cy = this.DOM / 2;

    // ── tanh格子伸縮: 非均一格子面座標を事前計算 ─────────────────────
    // x,y: ノズル軸(cx=cy=1.0m)に向かって集中 (BETA_XY=1.8)
    // z:  室内は天井(z=DOM)付近を密(betaT)、床側をやや密(betaF)
    //
    // 【重要 / GPUモードとの整合】
    //   旧版は「室内+ノズル」を合わせて z∈[0, DOM=2.0] に押し込み、
    //   描画時に _toWorldY() で z<1.82 を -1..1 に、z>=1.82 を 1..1.18 に
    //   マッピングしていた。この結果
    //     ・室内が 2.0m ではなく 1.82m しか無く、立方体にならない
    //     ・ノズル軸から外れた周縁部のセル(z: 1.82→2.0)まで天井より上
    //       (y: 1.0→1.18)に描かれ、上面に向かって伸びた形状に見える
    //   という不具合が出ていた。
    //
    //   本版は webgpu_mps.js(GPUモード)と同じ構成に揃える:
    //     室内     z ∈ [0, DOM]        → ワールド y ∈ [-1, +1] (厳密な立方体)
    //     ノズル   z ∈ [DOM, DOM_Z]    → ワールド y ∈ [+1, +1.18] (天井の上に直立)
    //   ノズル区間はセル層を「追加」する形にし、室内格子を一切削らない。
    const BETA_XY = 1.8;
    const BETA_ZT = 2.5;  // 天井(ノズル出口)側
    const BETA_ZF = 1.2;  // 床側
    this.xg = buildStretchedGrid(NX + 1, this.DOM, cx, BETA_XY);
    this.yg = buildStretchedGrid(NX + 1, this.DOM, cy, BETA_XY);

    // z: 室内 NX 層 (0→DOM, tanh伸縮) + ノズル NZ_NOZ 層 (DOM→DOM_Z, 等間隔)
    const roomZ = buildStretchedZGrid(NX + 1, this.DOM, BETA_ZT, BETA_ZF);
    const NZN   = this.NZ_NOZ;
    const NZ    = NX + NZN;
    this.NZ     = NZ;
    const zg = new Float64Array(NZ + 1);
    for (let k = 0; k <= NX; k++) zg[k] = roomZ[k];
    zg[NX] = this.DOM;  // 天井面をぴったり z=DOM に固定 (丸め誤差対策)
    for (let k = 1; k <= NZN; k++) zg[NX + k] = this.DOM + this.NOZZLE_H * k / NZN;
    this.zg = zg;

    const N = NX * NX * NZ;

    this.px   = new Float32Array(N); this.py   = new Float32Array(N);
    this.pz   = new Float32Array(N); this.vx   = new Float32Array(N);
    this.vy   = new Float32Array(N); this.vz   = new Float32Array(N);
    this.vxs  = new Float32Array(N); this.vys  = new Float32Array(N);
    this.vzs  = new Float32Array(N); this.press= new Float32Array(N);
    this.temp = new Float32Array(N);
    this.k_turb   = new Float32Array(N).fill(1e-4);
    this.eps_turb = new Float32Array(N).fill(1e-4);
    this.nu_t     = new Float32Array(N);
    this.ptype= new Uint8Array(N);
    this.inletFrac = new Float32Array(N);
    this._divBuf = new Float32Array(N);
    this._pBuf   = new Float32Array(N);
    this._tBuf   = new Float32Array(N);

    // ── ノズル断面プロファイル(実寸) ────────────────────────────────
    // 実寸(等倍)のまま扱い、ノズル内部〜近傍セルだけをサブグリッドで
    // 細分サンプリングして開口率(inletFrac)を求める(_sampleNozzleCell)。
    const NOZZLE_H = this.NOZZLE_H;
    const zBot     = this.DOM;   // ノズル下端(室内側出口) = 天井面
    let sawInlet = false;

    for (let i = 0; i < NX; i++) {
    for (let j = 0; j < NX; j++) {
    for (let k = 0; k < NZ; k++) {
      const c = this._idx(i, j, k);
      // セル中心: 格子面座標の中点
      const x = (this.xg[i] + this.xg[i + 1]) * 0.5;
      const y = (this.yg[j] + this.yg[j + 1]) * 0.5;
      const z = (this.zg[k] + this.zg[k + 1]) * 0.5;
      this.px[c] = x; this.py[c] = y; this.pz[c] = z;

      // セル局所幅 (サブサンプリング用等価格子幅)
      const lhX = this.xg[i + 1] - this.xg[i];
      const lhY = this.yg[j + 1] - this.yg[j];
      const lhZ = this.zg[k + 1] - this.zg[k];
      const localH = (lhX + lhY + lhZ) / 3;

      const isSideWall  = (i === 0 || i === NX - 1 || j === 0 || j === NX - 1);
      const inNozzleBand = (k >= NX);          // z > DOM: ノズル区間
      const isCeilLayer  = (k === NX - 1);     // 室内最上層 = 天井層

      let type = this.FLUID, vz0 = 0, t0 = this.T_REF, fracHere = 0;

      if (inNozzleBand) {
        // ── ノズル区間 (天井の上) ──────────────────────────────────
        // 側壁列であってもノズル半径外なら VOID(非存在)にする。
        const zNoz = Math.min(NOZZLE_H, Math.max(0, z - zBot));
        const frac = this._sampleNozzleCell(nozzleType, nozzle, core, cx, cy, x, y, z, localH, zBot, NOZZLE_H);

        if (isSideWall || frac.outsideFrac > 0.98) {
          // ノズル胴体の外側 = 何も存在しない領域。描画しない。
          type = this.VOID;
        } else if (frac.inletFrac > 0.02) {
          const localV = this._nozzleLocalVelocity(nozzleType, nozzle, core, zNoz);
          if (k === NZ - 1) {
            // ノズル最上面 = AC接続側の流入境界 (Dirichlet)
            type = this.INLET;
            vz0  = -localV * frac.inletFrac;
            t0   = this.T_IN;
            fracHere = frac.inletFrac;
            sawInlet = true;
          } else {
            // ノズル内部は自由流体。テーパー形状は WALL セルが規定する。
            type = this.FLUID;
            vz0  = -localV * frac.inletFrac;
            t0   = this.T_IN;
            fracHere = frac.inletFrac;
          }
        } else {
          type = this.WALL;  // ノズル胴体(テーパー壁)または中心シリンダー
        }
      } else if (isSideWall) {
        type = this.WALL;
      } else if (k === 0) {
        type = this.WALL;   // 床: no-slip等温壁
      } else if (isCeilLayer) {
        // ── 天井層 ────────────────────────────────────────────────
        // ノズル出口断面(zNoz=0)でサンプリングし、開口部だけを流体にする。
        // 開口部はすぐ上のノズル内部セルと連続し、噴流がそのまま室内へ入る。
        const frac = this._sampleNozzleCell(nozzleType, nozzle, core, cx, cy, x, y, z, localH, zBot, NOZZLE_H, 0);
        if (frac.inletFrac > 0.02) {
          type = this.FLUID;
          t0   = this.T_IN;
          fracHere = frac.inletFrac;
          vz0  = -this._nozzleLocalVelocity(nozzleType, nozzle, core, 0) * frac.inletFrac;
        } else {
          type = this.WALL;  // 天井壁面
        }
      } else {
        // 部屋の初期温度は34℃で一様、微小な初期擾乱を与える
        t0  = this.T_REF;
        vz0 = (Math.random() - 0.5) * 0.02;
      }

      this.vx[c] = 0; this.vy[c] = 0; this.vz[c] = vz0;
      this.temp[c] = t0;
      this.press[c] = (type === this.INLET ? (this.inletPressure ?? 60.0) : 0);
      this.ptype[c] = type;
      this.inletFrac[c] = fracHere;
    }}}
    this.N = N;
    if (!sawInlet) {
      console.warn('[CPUFallbackMPS] ノズル' + nozzleType + ': 開口セルが検出されませんでした。NOZZLE_SUBSAMPLEを上げてください。');
    }
  }

  // ────────────────────────────────────────────────────────────────
  //  連続の式によるノズル局所軸方向速度 [m/s]
  //  (GPUモード webgpu_mps.js と同一の式)
  // ────────────────────────────────────────────────────────────────
  _nozzleLocalVelocity(nozzleType, nozzle, core, zNoz) {
    const sec = getNozzleSection(nozzleType, zNoz);
    const inner = getNozzleInnerStructure(nozzleType, zNoz);
    let blockedArea = 0;
    if (inner.hasInner) {
      blockedArea = Math.PI * (inner.rOut * inner.rOut - inner.rIn * inner.rIn);
    }
    const flowArea = Math.max(Math.PI * (sec.rInner * sec.rInner) - blockedArea, 1e-8);
    const inletArea = (typeof AC.inletArea === 'number' && AC.inletArea > 0)
      ? AC.inletArea
      : Math.PI * AC.inletRadius * AC.inletRadius;
    return Math.min(AC.inletVelocity * inletArea / flowArea, 20);
  }

  // ────────────────────────────────────────────────────────────────
  //  ノズル内部〜近傍セルの埋め込みサブグリッド(実寸ジオメトリ)
  //  格子解像度は上げずに、ノズル形状判定の実効解像度だけを
  //  NOZZLE_SUBSAMPLE^2 × 3高さ層 のサンプル数で引き上げる。
  //  返り値は各カテゴリの占有率(0〜1、合計1)。
  // ────────────────────────────────────────────────────────────────
  _sampleNozzleCell(nozzleType, nozzle, core, cx, cy, x, y, z, h, zBot, NOZZLE_H, zProfileFixed) {
    const SUB = this.NOZZLE_SUBSAMPLE;
    const half = h / 2;
    let nInlet = 0, nWallBody = 0, nOutside = 0, total = 0;
    for (let zi = 0; zi < 3; zi++) {
      const sz = z + (zi - 1) * (half * 0.9); // セル下部・中央・上部の3層
      // zProfileFixed を渡した場合はその断面(通常は出口 zNoz=0)で固定サンプリング
      const zProfile = (zProfileFixed !== undefined)
        ? zProfileFixed
        : Math.min(NOZZLE_H, Math.max(0, sz - zBot));
      const { rOuter } = getNozzleSection(nozzleType, zProfile);
      for (let sxi = 0; sxi < SUB; sxi++) {
        const sx = x - half + (sxi + 0.5) / SUB * h;
        for (let syi = 0; syi < SUB; syi++) {
          const sy = y - half + (syi + 0.5) / SUB * h;
          const dx = sx - cx, dy = sy - cy;
          const r = Math.sqrt(dx * dx + dy * dy);
          total++;
          if (r > rOuter) nOutside++;
          else if (isNozzleWall(nozzleType, zProfile, r)) nWallBody++;
          else nInlet++;
        }
      }
    }
    return { inletFrac: nInlet / total, wallFrac: nWallBody / total, outsideFrac: nOutside / total };
  }

  // ────────────────────────────────────────────────────────────────
  //  Three.js シーン初期化
  // ────────────────────────────────────────────────────────────────
  _initThree(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0d0d1a, 1);
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);

    this.scene = new THREE.Scene();

    // 透視カメラ
    const aspect = canvas.clientWidth / canvas.clientHeight;
    this.camera = new THREE.PerspectiveCamera(55, aspect, 0.1, 100);
    this.camera.position.set(0, 0.8, 4.5);
    this.camera.lookAt(0, 0, 0);

    // 照明
    const amb  = new THREE.AmbientLight(0xffffff, 0.4);
    const dir  = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(1, 2, 2);
    this.scene.add(amb, dir);

    // 部屋ワイヤーフレーム
    const boxGeo = new THREE.BoxGeometry(2, 2, 2);
    const edges  = new THREE.EdgesGeometry(boxGeo);
    const lineMat= new THREE.LineBasicMaterial({ color: 0x445577, transparent: true, opacity: 0.5 });
    this.scene.add(new THREE.LineSegments(edges, lineMat));

    // 部屋の半透明面 (天井・床)
    const planeMat = new THREE.MeshBasicMaterial({ color: 0x223344, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), planeMat.clone());
    floor.rotation.x = -Math.PI / 2; floor.position.y = -1;
    const ceil  = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), planeMat.clone());
    ceil.rotation.x = Math.PI / 2;  ceil.position.y =  1;
    this.scene.add(floor, ceil);

    // ノズルメッシュ
    this._updateNozzleMesh();

    // セル中心の点群
    this._initParticlePoints();
    this._initSectionSurface();
    this._initVectorLines();

    // リサイズ対応
    const resizeObs = new ResizeObserver(() => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      this.renderer.setSize(w, h);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    });
    resizeObs.observe(canvas);
  }

  _initOrbitControls() {
    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.target.set(0, 0, 0);
      this._initCtrlPan();
    }
  }

  _initCtrlPan() {
    let pan = null;
    const canvas = this.renderer.domElement;
    canvas.addEventListener('mousedown', e => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      e.stopPropagation();
      pan = { x: e.clientX, y: e.clientY, position: this.camera.position.clone(), target: this.controls.target.clone() };
      this.controls.enabled = false;
    }, true);
    canvas.addEventListener('mousemove', e => {
      if (!pan) return;
      e.preventDefault();
      e.stopPropagation();
      const dx = (e.clientX - pan.x) * 0.005;
      const dy = (e.clientY - pan.y) * 0.005;
      const offset = new THREE.Vector3(-dx, dy, 0);
      this.camera.position.copy(pan.position).add(offset);
      this.controls.target.copy(pan.target).add(offset);
    }, true);
    window.addEventListener('mouseup', () => {
      if (pan) this.controls.enabled = true;
      pan = null;
    });
  }

  // ────────────────────────────────────────────────────────────────
  //  セル・ワイヤーフレーム(立方体エッジ)Three.js オブジェクト初期化
  //  格子点を点群(スプライト)で強調する表示は、遠近投影下で規則格子の
  //  点が視線方向に重なって放射状の縞(モアレ)として見えてしまうため、
  //  セルごとに立方体の辺(ワイヤーフレーム)を描く方式に変更。
  // ────────────────────────────────────────────────────────────────
  _initParticlePoints() {
    if (this.points) { this.scene.remove(this.points); this.geometry.dispose(); }

    const N = this.N;

    // ── セルあたり最大9辺(18頂点) ────────────────────────────────────
    // 旧版はセルごとに立方体12辺すべてを描いていたため、隣り合うセルで
    // 同じ辺が4回重なって描かれ、線が太く濁って「箱が個々に浮いている」
    // 見た目になっていた。
    // 本版は各セルが「最小コーナーから出る3辺」だけを描く方式に変更する。
    //   ・隣接セルの辺と1本に繋がるため、格子全体が連続したラティスになる
    //   ・辺の重複描画が無くなり、線が均一な太さで整った格子に見える
    //   ・頂点数も 24→18 (実質的な描画本数は 12→3) に削減される
    // ドメイン端のセルだけは閉じるために最大6辺を追加する(合計9辺)。
    this.EDGES_PER_CELL = 9;

    const geo = new THREE.BufferGeometry();
    const V   = this.EDGES_PER_CELL * 2;          // 18頂点/セル
    const pos = new Float32Array(N * V * 3);
    const col = new Float32Array(N * V * 3);
    pos.fill(999);

    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(col, 3));

    const mat = new THREE.LineBasicMaterial({
      vertexColors  : true,
      transparent   : true,
      opacity       : 0.55,
      depthWrite    : false,
    });

    this.geometry = geo;
    this.points   = new THREE.LineSegments(geo, mat);
    this.scene.add(this.points);
  }

  _updateNozzleMesh() {
    if (this.nozzleMesh) this.scene.remove(this.nozzleMesh);
    const mesh = createNozzleMesh(this.nozzleType);
    // ノズル下端(室内側出口、y=0)を部屋の天井(Three.js y=1.0)に接続し、
    // ノズル本体は天井の上(y: 1.0〜1.18)に直立配置する。
    // 部屋の箱(2x2x2m)の天井は y=1.0 であり、部屋がノズル上面まで伸びないようにする。
    mesh.position.set(0, 1.0, 0);
    this.nozzleMesh = mesh;
    this.scene.add(mesh);
  }

  // ────────────────────────────────────────────────────────────────
  //  断面(セル)サーフェス初期化
  //  旧版は 25×25 の一様補間グリッドだったため、非均一格子の実セルと
  //  対応せず、粗い領域で格子との間に隙間・ズレが見えていた。
  //  本版は「実際の格子セル1個 = 四角形1枚(三角形2枚)」とし、頂点を
  //  格子面座標 (xg/yg/zg) にスナップさせるので断面が隙間なく埋まる。
  //  セル境界は別途 LineSegments で重ね描きして格子フレームワークにする。
  // ────────────────────────────────────────────────────────────────
  _initSectionSurface() {
    if (this.sectionSurface) {
      this.scene.remove(this.sectionSurface);
      this.sectionSurfaceGeometry?.dispose();
    }
    if (this.sectionGridLines) {
      this.scene.remove(this.sectionGridLines);
      this.sectionGridGeometry?.dispose();
    }

    const cells = this.NX * this.NZ;      // 面内: 水平NX × 鉛直NZ
    this.sectionCellCount = cells;

    // ── 面(塗り) ────────────────────────────────────────────────
    const vertices = new Float32Array(cells * 4 * 3);
    const colors   = new Float32Array(cells * 4 * 3);
    // 頂点数が 65,536 未満なら 16bit インデックスで十分 (WebGL1 互換)
    const indices  = (cells * 4 < 65536)
      ? new Uint16Array(cells * 6)
      : new Uint32Array(cells * 6);
    for (let c = 0; c < cells; c++) {
      const v = c * 4, o = c * 6;
      indices[o]     = v;     indices[o + 1] = v + 1; indices[o + 2] = v + 2;
      indices[o + 3] = v;     indices[o + 4] = v + 2; indices[o + 5] = v + 3;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    this.sectionSurfaceGeometry = geo;
    this.sectionSurface = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      transparent: false,   // 隙間・透け防止のため不透明で塗る
      depthWrite: true,
    }));
    this.sectionSurface.visible = false;
    this.scene.add(this.sectionSurface);

    // ── セル境界線 (格子フレームワーク) ─────────────────────────
    // 1セルあたり4辺 = 8頂点。隣接セルと重複するが、頂点が完全に一致する
    // ため線が二重に太く見えることはない。
    const linePos = new Float32Array(cells * 8 * 3);
    linePos.fill(999);
    const lgeo = new THREE.BufferGeometry();
    lgeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
    this.sectionGridGeometry = lgeo;
    this.sectionGridLines = new THREE.LineSegments(lgeo, new THREE.LineBasicMaterial({
      color: 0x0e1520,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    }));
    this.sectionGridLines.visible = false;
    this.scene.add(this.sectionGridLines);
  }

  _initVectorLines() {
    if (this.vectorLines) {
      this.scene.remove(this.vectorLines);
      this.vectorGeometry?.dispose();
    }
    const maxV = this.maxVectors;
    const positions = new Float32Array(maxV * 6 * 3);
    const colors    = new Float32Array(maxV * 6 * 3);
    // 初期値は描画外へ
    positions.fill(999);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
    });

    this.vectorGeometry = geo;
    this.vectorLines = new THREE.LineSegments(geo, mat);
    this.vectorLines.visible = false;
    this.scene.add(this.vectorLines);
  }

  // ────────────────────────────────────────────────────────────────
  //  断面(セル)サーフェス更新
  //  断面位置 slicePosition を含むセル列を1枚ずつ、実セル矩形として描く。
  //  補間は行わず、セル値をそのまま塗る(フラットシェーディング)ので
  //  FVMの解がどのセルの値なのかがそのまま読み取れる。
  // ────────────────────────────────────────────────────────────────
  _updateSectionSurface() {
    if (!this.sectionSurfaceGeometry) return;
    const show = (this.visualMode !== 0);
    this.sectionSurface.visible  = show && (this.displayMode !== 1);
    this.sectionGridLines.visible = show;
    if (!show) return;

    const NX = this.NX, NZ = this.NZ;
    const positions = this.sectionSurfaceGeometry.attributes.position.array;
    const colors    = this.sectionSurfaceGeometry.attributes.color.array;
    const linePos   = this.sectionGridGeometry.attributes.position.array;

    const yz = (this.sliceMode === 1);          // true: YZ断面(x固定)
    // 断面が属するセル添字 (非均一格子なので二分探索)
    const fIdx = yz ? bisectLeft(this.xg, this.slicePosition)
                    : bisectLeft(this.yg, this.slicePosition);
    const hg   = yz ? this.yg : this.xg;        // 面内水平方向の格子面座標
    const fixedWorld = this.slicePosition - 1.0;

    let li = 0;   // 線分頂点カウンタ
    for (let b = 0; b < NZ; b++) {
      const y0 = this._toWorldY(this.zg[b]);
      const y1 = this._toWorldY(this.zg[b + 1]);
      for (let a = 0; a < NX; a++) {
        const cellId = b * NX + a;
        const base   = cellId * 12;             // 4頂点 × 3成分
        const c      = yz ? this._idx(fIdx, a, b) : this._idx(a, fIdx, b);
        const type   = this.ptype[c];

        // VOIDセル(ノズル胴体より外側の非存在セル)は描かない
        if (type === this.VOID) {
          for (let v = 0; v < 12; v++) positions[base + v] = 999;
          continue;
        }

        const h0 = hg[a] - 1.0, h1 = hg[a + 1] - 1.0;

        // ワールド座標: YZ断面 → (x固定, y=z, z=面内h) / XZ断面 → (x=面内h, y=z, z固定)
        const px0 = yz ? fixedWorld : h0;
        const px1 = yz ? fixedWorld : h1;
        const pz0 = yz ? h0 : fixedWorld;
        const pz1 = yz ? h1 : fixedWorld;

        // 反時計回り: (h0,y0) (h1,y0) (h1,y1) (h0,y1)
        positions[base]      = px0; positions[base + 1]  = y0; positions[base + 2]  = pz0;
        positions[base + 3]  = px1; positions[base + 4]  = y0; positions[base + 5]  = pz1;
        positions[base + 6]  = px1; positions[base + 7]  = y1; positions[base + 8]  = pz1;
        positions[base + 9]  = px0; positions[base + 10] = y1; positions[base + 11] = pz0;

        // ── セル値 → 色 (セル内は一様なフラット塗り) ──────────────
        let r, g, bl;
        if (type === this.WALL) {
          r = 0.20; g = 0.23; bl = 0.30;        // 壁も塗って穴を作らない
        } else {
          let value = this.temp[c];
          if (this.displayMode === 1) {
            value = Math.sqrt(this.vx[c] ** 2 + this.vy[c] ** 2 + this.vz[c] ** 2);
          } else if (this.displayMode === 2) {
            value = Math.abs(this.press[c]);
          }
          let t;
          if (this.displayMode === 0) {
            t = (value - this.T_IN) / (this.T_REF - this.T_IN);
          } else if (this.displayMode === 1) {
            t = value / 15;
          } else {
            const pScale = Math.max(40.0, (this.inletPressure ?? 60.0) * 1.2);
            t = value / pScale;
          }
          [r, g, bl] = jetColorJS(Math.max(0, Math.min(1, t)));
        }
        for (let v = 0; v < 4; v++) {
          const o = base + v * 3;
          colors[o] = r; colors[o + 1] = g; colors[o + 2] = bl;
        }

        // ── セル境界線 (4辺) ──────────────────────────────────────
        const put = (ax, ay, az, bx, by, bz) => {
          linePos[li]     = ax; linePos[li + 1] = ay; linePos[li + 2] = az;
          linePos[li + 3] = bx; linePos[li + 4] = by; linePos[li + 5] = bz;
          li += 6;
        };
        put(px0, y0, pz0, px1, y0, pz1);
        put(px1, y0, pz1, px1, y1, pz1);
        put(px1, y1, pz1, px0, y1, pz0);
        put(px0, y1, pz0, px0, y0, pz0);
      }
    }
    for (let v = li; v < linePos.length; v++) linePos[v] = 999;

    this.sectionSurfaceGeometry.attributes.position.needsUpdate = true;
    this.sectionSurfaceGeometry.attributes.color.needsUpdate    = true;
    this.sectionGridGeometry.attributes.position.needsUpdate    = true;
  }

  // ────────────────────────────────────────────────────────────────
  //  格子インデックス・境界参照ヘルパー
  // ────────────────────────────────────────────────────────────────
  _idx(i, j, k) {
    const NX = this.NX;
    return i + NX * (j + NX * k);
  }

  /**
   * 任意のワールド座標(x,y,z)が属する格子セルの ptype を返す。
   * 断面図(_updateSectionSurface)でノズル胴(WALL)を塗り分けるために使用。
   * 非均一格子対応: bisectLeft で格子面配列を二分探索してセル添字を逆引き。
   */
  _ptypeAt(x, y, z) {
    const NX = this.NX;
    const i = bisectLeft(this.xg, x);
    const j = bisectLeft(this.yg, y);
    const k = bisectLeft(this.zg, z);
    return this.ptype[this._idx(i, j, k)];
  }


  /**
   * 境界を考慮したフィールド参照。
   * - i,j は常に有効範囲(側壁セルが実在配列要素として存在するため)
   * - k<0 は本来到達しない(床(k=0)がno-slip固体壁として実セルで
   *   存在し、流体セルはk=1以上のみのため、k=0近傍の勾配計算は
   *   常に実在の床WALLセルを参照できる)。範囲外アクセス防止のため
   *   クランプのみ残す
   * - k>NX-1 は原理上発生しない(天井層は実セルとして常に存在)ためクランプのみ
   */
  _at(field, i, j, k) {
    const NX = this.NX, NZ = this.NZ;
    if (i < 0) i = 0; else if (i > NX - 1) i = NX - 1;
    if (j < 0) j = 0; else if (j > NX - 1) j = NX - 1;
    if (k < 0) k = 0; else if (k > NZ - 1) k = NZ - 1;
    return field[this._idx(i, j, k)];
  }

  // ────────────────────────────────────────────────────────────────
  //  1ステップ計算 (非均一格子 FVM: 移流(風上)+拡散(2次精度非均一)+浮力 → 投影法)
  //
  //  非均一差分の定式化 (2次精度、非対称ステンシル):
  //    df/dx ≈ (fE - fW) / (hE + hW)  ... 1次導関数 (中心差分相当)
  //    d²f/dx² ≈ 2*(fE/hE - f*(1/hE+1/hW) + fW/hW) / (hE+hW)  ... ラプラシアン
  //  圧力ポアソン(ヤコビ法)の係数:
  //    ∇²p = Σ 2*(pNeighbor - p) / (hNeighbor*(hNeighbor+hOpposite)) = rhs
  //    → 非均一な係数で各方向を重み付け平均
  // ────────────────────────────────────────────────────────────────
  step() {
    const NX = this.NX, NZ = this.NZ, DT = this.DT;
    const RHO = this.RHO0, NU = this.NU, ALPHA = this.ALPHA;
    const G = this.G, BETA = this.BETA, TREF = this.T_REF;
    const xg = this.xg, yg = this.yg, zg = this.zg;

    // ── 1. 運動量方程式: 仮速度 u* (圧力項を除く) ────────────────────
    for (let i = 1; i < NX - 1; i++) {
    for (let j = 1; j < NX - 1; j++) {
    for (let k = 0; k < NZ - 1; k++) {
      const c = this._idx(i, j, k);
      if (this.ptype[c] !== this.FLUID) {
        this.vxs[c] = this.vx[c]; this.vys[c] = this.vy[c]; this.vzs[c] = this.vz[c];
        continue;
      }
      const u = this.vx[c], v = this.vy[c], w = this.vz[c], T = this.temp[c];

      // 局所格子幅 (格子面配列から算出)
      const hxE = xg[i+1] - xg[i],  hxW = xg[i] - xg[i-1];
      const hyN = yg[j+1] - yg[j],  hyS = yg[j] - yg[j-1];
      const hzU = zg[k+1] - zg[k],  hzD = (k > 0) ? zg[k] - zg[k-1] : zg[k+1] - zg[k];

      const uE=this._at(this.vx,i+1,j,k), uW=this._at(this.vx,i-1,j,k);
      const uN=this._at(this.vx,i,j+1,k), uS=this._at(this.vx,i,j-1,k);
      const uU=this._at(this.vx,i,j,k+1), uD=this._at(this.vx,i,j,k-1);
      const vE=this._at(this.vy,i+1,j,k), vW=this._at(this.vy,i-1,j,k);
      const vN=this._at(this.vy,i,j+1,k), vS=this._at(this.vy,i,j-1,k);
      const vU=this._at(this.vy,i,j,k+1), vD=this._at(this.vy,i,j,k-1);
      const wE=this._at(this.vz,i+1,j,k), wW=this._at(this.vz,i-1,j,k);
      const wN=this._at(this.vz,i,j+1,k), wS=this._at(this.vz,i,j-1,k);
      const wU=this._at(this.vz,i,j,k+1), wD=this._at(this.vz,i,j,k-1);

      // 風上差分 (donor-cell): 非均一格子版
      // 風上側の局所格子幅を使い、風下 vs 風上の逆格子幅で規格化
      const dudx = u>=0 ? (u-uW)/hxW : (uE-u)/hxE;
      const dudy = v>=0 ? (u-uS)/hyS : (uN-u)/hyN;
      const dudz = w>=0 ? (u-uD)/hzD : (uU-u)/hzU;
      const dvdx = u>=0 ? (v-vW)/hxW : (vE-v)/hxE;
      const dvdy = v>=0 ? (v-vS)/hyS : (vN-v)/hyN;
      const dvdz = w>=0 ? (v-vD)/hzD : (vU-v)/hzU;
      const dwdx = u>=0 ? (w-wW)/hxW : (wE-w)/hxE;
      const dwdy = v>=0 ? (w-wS)/hyS : (wN-w)/hyN;
      const dwdz = w>=0 ? (w-wD)/hzD : (wU-w)/hzU;

      // 拡散項: 非均一格子2次精度ラプラシアン
      //   d²f/dx² ≈ 2*(fE/hxE - f*(1/hxE+1/hxW) + fW/hxW) / (hxE+hxW)
      const lapU = 2.0 * ((uE/hxE - u*(1/hxE+1/hxW) + uW/hxW) / (hxE+hxW)
                        + (uN/hyN - u*(1/hyN+1/hyS) + uS/hyS) / (hyN+hyS)
                        + (uU/hzU - u*(1/hzU+1/hzD) + uD/hzD) / (hzU+hzD));
      const lapV = 2.0 * ((vE/hxE - v*(1/hxE+1/hxW) + vW/hxW) / (hxE+hxW)
                        + (vN/hyN - v*(1/hyN+1/hyS) + vS/hyS) / (hyN+hyS)
                        + (vU/hzU - v*(1/hzU+1/hzD) + vD/hzD) / (hzU+hzD));
      const lapW = 2.0 * ((wE/hxE - w*(1/hxE+1/hxW) + wW/hxW) / (hxE+hxW)
                        + (wN/hyN - w*(1/hyN+1/hyS) + wS/hyS) / (hyN+hyS)
                        + (wU/hzU - w*(1/hzU+1/hzD) + wD/hzD) / (hzU+hzD));

      const buoy = BETA * (T - TREF) * G;  // Boussinesq浮力項

      this.vxs[c] = u + DT * (-(u*dudx + v*dudy + w*dudz) + NU*lapU);
      this.vys[c] = v + DT * (-(u*dvdx + v*dvdy + w*dvdz) + NU*lapV);
      this.vzs[c] = w + DT * (-(u*dwdx + v*dwdy + w*dwdz) + NU*lapW - G + buoy);
    }}}

    // 速度クランプ (数値発散防止、可視化上の上限)
    for (let c = 0; c < this.N; c++) {
      if (this.ptype[c] !== this.FLUID) continue;
      const vlen = Math.hypot(this.vxs[c], this.vys[c], this.vzs[c]);
      if (vlen > 20) { const s = 20 / vlen; this.vxs[c]*=s; this.vys[c]*=s; this.vzs[c]*=s; }
    }

    // ── 2. 発散計算 (圧力ポアソン方程式の右辺) ────────────────────────
    // 非均一格子の中心差分: ∂u/∂x ≈ (u_{i+1}-u_{i-1})/(hxE+hxW)
    const div = this._divBuf;
    div.fill(0);
    for (let i = 1; i < NX - 1; i++) {
    for (let j = 1; j < NX - 1; j++) {
    for (let k = 1; k < NZ - 1; k++) {
      const c = this._idx(i, j, k);
      if (this.ptype[c] !== this.FLUID) continue;
      const hxE = xg[i+1]-xg[i], hxW = xg[i]-xg[i-1];
      const hyN = yg[j+1]-yg[j], hyS = yg[j]-yg[j-1];
      const hzU = zg[k+1]-zg[k], hzD = zg[k]-zg[k-1];
      const dudx = (this._at(this.vxs,i+1,j,k) - this._at(this.vxs,i-1,j,k)) / (hxE+hxW);
      const dvdy = (this._at(this.vys,i,j+1,k) - this._at(this.vys,i,j-1,k)) / (hyN+hyS);
      const dwdz = (this._at(this.vzs,i,j,k+1) - this._at(this.vzs,i,j,k-1)) / (hzU+hzD);
      div[c] = (dudx + dvdy + dwdz) * RHO / DT;
    }}}

    // ── 3. 圧力ポアソン方程式 ∇²p = div  (非均一格子 ヤコビ法) ──────────
    // 非均一格子の ∇²p: 各方向の寄与を 2/(h_neighbor * (h_E+h_W)) の係数で重み付け。
    // Dirichlet BC: WALL → p=0, INLET → p=inletPressure (冷風扇加圧)
    const pInlet = this.inletPressure ?? 60.0;
    for (let c = 0; c < this.N; c++) {
      if (this.ptype[c] === this.INLET) this.press[c] = pInlet;
      else if (this.ptype[c] === this.WALL) this.press[c] = 0;
    }

    const pNew = this._pBuf;
    pNew.fill(0);
    for (let it = 0; it < this.PRESSURE_ITERS; it++) {
      for (let i = 1; i < NX - 1; i++) {
      for (let j = 1; j < NX - 1; j++) {
      for (let k = 1; k < NZ - 1; k++) {
        const c = this._idx(i, j, k);
        if (this.ptype[c] === this.INLET) { pNew[c] = pInlet; continue; }
        if (this.ptype[c] !== this.FLUID) { pNew[c] = 0; continue; }

        // 局所格子幅
        const hxE = xg[i+1]-xg[i], hxW = xg[i]-xg[i-1];
        const hyN = yg[j+1]-yg[j], hyS = yg[j]-yg[j-1];
        const hzU = zg[k+1]-zg[k], hzD = zg[k]-zg[k-1];

        // ラプラシアンの各方向係数: a_E=2/(hxE*(hxE+hxW)) など
        const aE = 2.0/(hxE*(hxE+hxW)), aW = 2.0/(hxW*(hxE+hxW));
        const aN = 2.0/(hyN*(hyN+hyS)), aS = 2.0/(hyS*(hyN+hyS));
        const aU = 2.0/(hzU*(hzU+hzD)), aD = 2.0/(hzD*(hzU+hzD));
        const aC = aE + aW + aN + aS + aU + aD;

        const pE=this._at(this.press,i+1,j,k), pW=this._at(this.press,i-1,j,k);
        const pN=this._at(this.press,i,j+1,k), pS=this._at(this.press,i,j-1,k);
        const pU=this._at(this.press,i,j,k+1), pD=this._at(this.press,i,j,k-1);
        pNew[c] = (aE*pE + aW*pW + aN*pN + aS*pS + aU*pU + aD*pD - div[c]) / aC;
      }}}
      this.press.set(pNew);
    }

    // ── 4. 速度補正 (投影): u = u* − (Δt/ρ)∇p ─────────────────────────
    // 非均一格子の中心差分: ∂p/∂x ≈ (p_{i+1}-p_{i-1})/(hxE+hxW)
    for (let i = 1; i < NX - 1; i++) {
    for (let j = 1; j < NX - 1; j++) {
    for (let k = 0; k < NZ - 1; k++) {
      const c = this._idx(i, j, k);
      if (this.ptype[c] !== this.FLUID) continue;
      const hxE = xg[i+1]-xg[i], hxW = xg[i]-xg[i-1];
      const hyN = yg[j+1]-yg[j], hyS = yg[j]-yg[j-1];
      const hzU = zg[k+1]-zg[k], hzD = (k > 0) ? zg[k]-zg[k-1] : zg[k+1]-zg[k];
      const pE=this._at(this.press,i+1,j,k), pW=this._at(this.press,i-1,j,k);
      const pN=this._at(this.press,i,j+1,k), pS=this._at(this.press,i,j-1,k);
      const pU=this._at(this.press,i,j,k+1), pD=this._at(this.press,i,j,k-1);
      this.vx[c] = this.vxs[c] - DT/RHO * (pE - pW) / (hxE+hxW);
      this.vy[c] = this.vys[c] - DT/RHO * (pN - pS) / (hyN+hyS);
      this.vz[c] = this.vzs[c] - DT/RHO * (pU - pD) / (hzU+hzD);
    }}}

    // ── 5. 温度輸送 (移流(風上)+拡散、補正済み速度場を使用) ───────────
    const tempBuf = this._tBuf;
    tempBuf.set(this.temp);
    for (let i = 1; i < NX - 1; i++) {
    for (let j = 1; j < NX - 1; j++) {
    for (let k = 0; k < NZ - 1; k++) {
      const c = this._idx(i, j, k);
      if (this.ptype[c] !== this.FLUID) continue;
      const u = this.vx[c], v = this.vy[c], w = this.vz[c], T = this.temp[c];

      const hxE = xg[i+1]-xg[i], hxW = xg[i]-xg[i-1];
      const hyN = yg[j+1]-yg[j], hyS = yg[j]-yg[j-1];
      const hzU = zg[k+1]-zg[k], hzD = (k > 0) ? zg[k]-zg[k-1] : zg[k+1]-zg[k];

      const TE=this._at(this.temp,i+1,j,k), TW=this._at(this.temp,i-1,j,k);
      const TN=this._at(this.temp,i,j+1,k), TS=this._at(this.temp,i,j-1,k);
      const TU=this._at(this.temp,i,j,k+1), TD=this._at(this.temp,i,j,k-1);

      // 風上差分 (温度移流)
      const dTdx = u>=0 ? (T-TW)/hxW : (TE-T)/hxE;
      const dTdy = v>=0 ? (T-TS)/hyS : (TN-T)/hyN;
      const dTdz = w>=0 ? (T-TD)/hzD : (TU-T)/hzU;

      // ラプラシアン (熱拡散)
      const lapT = 2.0 * ((TE/hxE - T*(1/hxE+1/hxW) + TW/hxW) / (hxE+hxW)
                        + (TN/hyN - T*(1/hyN+1/hyS) + TS/hyS) / (hyN+hyS)
                        + (TU/hzU - T*(1/hzU+1/hzD) + TD/hzD) / (hzU+hzD));

      const Tn = T + DT * (-(u*dTdx + v*dTdy + w*dTdz) + ALPHA*lapT);
      tempBuf[c] = Math.max(this.T_IN - 1, Math.min(this.T_REF + 2, Tn));
    }}}
    this.temp.set(tempBuf);

    if (this.useTurbulence) {
      this._stepTurbulence();
    }

    this.stepCount++;
    this.stats.step = this.stepCount;
    this.stats.time = this.stepCount * this.DT;
  }


  // ────────────────────────────────────────────────────────────────
  //  Three.js レンダリング更新
  // ────────────────────────────────────────────────────────────────
  _toWorldY(z) {
    // GPUモードと同一の線形マッピング (webgpu_mps.js: y = z + DOM/2 - DOM = z - 1.0)。
    //   室内   z ∈ [0, 2.0]    → y ∈ [-1.0, +1.0]  … 2×2×2 の厳密な立方体
    //   ノズル z ∈ [2.0, 2.18] → y ∈ [+1.0, +1.18] … 天井の上に直立
    // 旧版のような区間ごとのスケール切替は行わない(室内が立方体にならず、
    // 周縁部のセルが天井より上に伸びて見える原因だった)。
    return z - this.DOM / 2;
  }

  _updateVisuals() {
    const N    = this.N;
    const NX   = this.NX;
    const NZ   = this.NZ;
    const pos  = this.geometry.attributes.position.array;
    const col  = this.geometry.attributes.color.array;
    const Tmin = this.T_IN, Tmax = this.T_REF;
    let tMin=Tmax, tMax=Tmin, vMax=0;

    // メッシュワイヤーフレームを描くのは 3D表示(visualMode===0) かつ
    // 速度表示以外のとき。断面/コンター表示では、線だけのセル表示だと
    // セル間に隙間が見えるため、_updateSectionSurface() の
    // 「塗り + セル境界線」による隙間なし格子に一本化する。
    const showMesh = (this.displayMode !== 1) && (this.visualMode === 0);
    this.points.visible = showMesh;

    const sliceThickness = this.visualMode === 2 ? this.H * 2.5 : this.H * 0.8;
    const V = this.EDGES_PER_CELL * 2;   // 18頂点/セル
    const STRIDE = V * 3;                // 54成分/セル

    if (showMesh) {
      const seg = new Float32Array(this.EDGES_PER_CELL * 6);

      for (let i = 0; i < N; i++) {
        const type   = this.ptype[i];
        const isWall = (type === this.WALL);
        const isVoid = (type === this.VOID);
        const base   = i * STRIDE;

        const sliceCoord = this.sliceMode === 1 ? this.px[i] : this.py[i];
        // VOIDセル(ノズル胴体より外側の非存在セル)は常に描画しない。
        // これで室内立方体の上面より上に余計な格子が出なくなる。
        const inView = !isVoid &&
          (this.visualMode === 0 || Math.abs(sliceCoord - this.slicePosition) <= sliceThickness);

        if (inView) {
          // セル添字の逆引き
          const k   = Math.floor(i / (NX * NX));
          const rem = i % (NX * NX);
          const j   = Math.floor(rem / NX);
          const ii  = rem % NX;

          // ワールド座標: X=xg-1, Z=yg-1, Y=_toWorldY(zg)
          const x0 = this.xg[ii] - 1.0,  x1 = this.xg[ii + 1] - 1.0;
          const z0 = this.yg[j]  - 1.0,  z1 = this.yg[j + 1]  - 1.0;
          const y0 = this._toWorldY(this.zg[k]);
          const y1 = this._toWorldY(this.zg[k + 1]);

          // ── 共有辺方式 ────────────────────────────────────────────
          // 最小コーナー(x0,y0,z0)から出る3辺のみを描く。隣接セルの
          // 対応する辺と端点が一致するので、格子全体が1本の連続した
          // ラティスとして繋がる(辺の重複描画なし)。
          let e = 0;
          const put = (ax, ay, az, bx, by, bz) => {
            const o = e * 6;
            seg[o]   = ax; seg[o+1] = ay; seg[o+2] = az;
            seg[o+3] = bx; seg[o+4] = by; seg[o+5] = bz;
            e++;
          };
          put(x0, y0, z0, x1, y0, z0);  // x方向
          put(x0, y0, z0, x0, y1, z0);  // y方向(鉛直)
          put(x0, y0, z0, x0, y0, z1);  // z方向

          // ドメイン端のセルは外側の面を閉じる
          if (ii === NX - 1) {
            put(x1, y0, z0, x1, y1, z0);
            put(x1, y0, z0, x1, y0, z1);
          }
          if (j === NX - 1) {
            put(x0, y0, z1, x1, y0, z1);
            put(x0, y0, z1, x0, y1, z1);
          }
          if (k === NZ - 1) {
            put(x0, y1, z0, x1, y1, z0);
            put(x0, y1, z0, x0, y1, z1);
          }

          for (let v = 0; v < e * 2; v++) {
            const o = base + v * 3, so = v * 3;
            pos[o] = seg[so]; pos[o + 1] = seg[so + 1]; pos[o + 2] = seg[so + 2];
          }
          for (let v = e * 2; v < V; v++) {
            const o = base + v * 3;
            pos[o] = 999; pos[o + 1] = 999; pos[o + 2] = 999;
          }
        } else {
          for (let v = 0; v < V; v++) {
            const o = base + v * 3;
            pos[o] = 999; pos[o + 1] = 999; pos[o + 2] = 999;
          }
        }

        let t;
        if (this.displayMode === 0) {
          t = (this.temp[i] - Tmin) / (Tmax - Tmin);
        } else {
          const pScale = Math.max(40.0, (this.inletPressure ?? 60.0) * 1.2);
          t = Math.min(Math.abs(this.press[i]) / pScale, 1);
        }
        t = Math.max(0, Math.min(1, t));
        const [r, g, b] = (!inView || isWall) ? [0.34, 0.38, 0.48] : jetColorJS(t);
        for (let v = 0; v < V; v++) {
          const o = base + v * 3;
          col[o] = r; col[o + 1] = g; col[o + 2] = b;
        }

        if (type === this.FLUID || type === this.INLET) {
          tMin = Math.min(tMin, this.temp[i]);
          tMax = Math.max(tMax, this.temp[i]);
          vMax = Math.max(vMax, Math.sqrt(this.vx[i]**2 + this.vy[i]**2 + this.vz[i]**2));
        }
      }

      this.geometry.attributes.position.needsUpdate = true;
      this.geometry.attributes.color.needsUpdate    = true;
    } else {
      // 速度表示モード時の統計集計
      for (let i = 0; i < N; i++) {
        const type = this.ptype[i];
        if (type === this.FLUID || type === this.INLET) {
          tMin = Math.min(tMin, this.temp[i]);
          tMax = Math.max(tMax, this.temp[i]);
          vMax = Math.max(vMax, Math.sqrt(this.vx[i]**2 + this.vy[i]**2 + this.vz[i]**2));
        }
      }
    }

    this.stats.tMin = tMin; this.stats.tMax = tMax; this.stats.vMax = vMax;
    this._updateSectionSurface();
    this._updateVectorLines();
  }

  // ────────────────────────────────────────────────────────────────
  //  1本の矢印(6頂点=3線分)を positions/colors バッファへ書き込む。
  //  呼び出し側は「1セルにつき1回だけ」呼ぶこと(重複防止の前提)。
  //  戻り値: 描画したら true (呼び出し側でvCountを1進める)、
  //          速度が閾値未満で描かなかったら false。
  // ────────────────────────────────────────────────────────────────
  _writeVectorArrow(c, vCount, positions, colors) {
    const u = this.vx[c], v = this.vy[c], w = this.vz[c];
    const speed = Math.sqrt(u * u + v * v + w * w);
    if (speed < 0.08) return false;

    // Three.js座標系: X = px-1.0, Y = _toWorldY(pz), Z = py-1.0
    // 速度ベクトル: dirX = u, dirY = w, dirZ = v
    const sx = this.px[c] - 1.0;
    const sy = this._toWorldY(this.pz[c]);
    const sz = this.py[c] - 1.0;

    const invS = 1.0 / speed;
    const dx = u * invS, dy = w * invS, dz = v * invS;

    const len = Math.min(0.085, Math.max(0.025, speed * 0.007));
    const ex = sx + dx * len;
    const ey = sy + dy * len;
    const ez = sz + dz * len;

    const hLen = len * 0.35;
    let ux = 0, uy = 1, uz = 0;
    if (Math.abs(dy) > 0.92) { ux = 1; uy = 0; uz = 0; }
    let sx_vec = dy * uz - dz * uy;
    let sy_vec = dz * ux - dx * uz;
    let sz_vec = dx * uy - dy * ux;
    const sLen = Math.hypot(sx_vec, sy_vec, sz_vec) || 1.0;
    sx_vec = (sx_vec / sLen) * (hLen * 0.55);
    sy_vec = (sy_vec / sLen) * (hLen * 0.55);
    sz_vec = (sz_vec / sLen) * (hLen * 0.55);

    const hAx = ex - dx * hLen + sx_vec;
    const hAy = ey - dy * hLen + sy_vec;
    const hAz = ez - dz * hLen + sz_vec;

    const hBx = ex - dx * hLen - sx_vec;
    const hBy = ey - dy * hLen - sy_vec;
    const hBz = ez - dz * hLen - sz_vec;

    // 流速スカラー量をベクトルの色に反映 (Jetカラー: 青→緑→黄→赤)
    const t = Math.max(0, Math.min(1, speed / 15.0));
    const [r, g, b] = jetColorJS(t);

    const baseP = vCount * 18;
    positions[baseP]     = sx;  positions[baseP + 1] = sy;  positions[baseP + 2] = sz;
    positions[baseP + 3] = ex;  positions[baseP + 4] = ey;  positions[baseP + 5] = ez;
    positions[baseP + 6] = ex;  positions[baseP + 7] = ey;  positions[baseP + 8] = ez;
    positions[baseP + 9] = hAx; positions[baseP + 10]= hAy; positions[baseP + 11]= hAz;
    positions[baseP + 12]= ex;  positions[baseP + 13]= ey;  positions[baseP + 14]= ez;
    positions[baseP + 15]= hBx; positions[baseP + 16]= hBy; positions[baseP + 17]= hBz;

    for (let vert = 0; vert < 6; vert++) {
      const cIdx = baseP + vert * 3;
      colors[cIdx]     = r;
      colors[cIdx + 1] = g;
      colors[cIdx + 2] = b;
    }
    return true;
  }

  // ────────────────────────────────────────────────────────────────
  //  速度ベクトル(矢印)レンダリング更新
  //  流速スカラー量(大きさ |u|)に応じたJetカラーマップを矢印の色に反映
  //
  //  【重複ベクトル対策】
  //    旧版は断面/コンター表示時にも法線方向(sliceMode=1ならi、=2ならj)
  //    をstride=1で全セル走査し、「slicePositionからの距離が
  //    sliceThickness未満」という“厚み”条件で通過セルを選んでいた。
  //    本ソルバーはtanh伸縮格子でノズル軸(中心)付近ほどセル幅が
  //    平均格子幅Hより小さくなるため、中心付近ではsliceThicknessの
  //    範囲内に法線方向の隣接セルが複数入ってしまい、断面上は同じ
  //    位置にしか見えない(j,k)に対して実質同じベクトルが2〜3本
  //    重ねて描かれていた(＝「1セルから複数ベクトルが重なる」不具合)。
  //    本版は_updateSectionSurfaceと同じ方式でslicePositionを含む
  //    セル添字を1つだけ二分探索(bisectLeft)で確定し、断面内の
  //    面内2方向だけを走査する。法線方向の添字が1つに固定されるため、
  //    断面の格子1マスにつきベクトルは必ず1本になる。
  // ────────────────────────────────────────────────────────────────
  _updateVectorLines() {
    if (!this.vectorGeometry || !this.vectorLines) return;

    const show = (this.displayMode === 1);
    this.vectorLines.visible = show;
    if (!show) return;

    const positions = this.vectorGeometry.attributes.position.array;
    const colors    = this.vectorGeometry.attributes.color.array;
    const maxV = this.maxVectors;
    let vCount = 0;

    const NX = this.NX;
    const isSlice = (this.visualMode !== 0);   // 断面 or コンター表示

    if (isSlice) {
      // ── 断面/コンター表示: slicePositionを含む1列のセルだけを対象に、
      //    面内2方向(YZ断面ならj,k / XZ断面ならi,k)をstride=1で走査。
      //    法線方向は固定添字1つに決まるので、1マス=1ベクトルになる。
      const yz = (this.sliceMode === 1);
      const fIdx = yz ? bisectLeft(this.xg, this.slicePosition)
                      : bisectLeft(this.yg, this.slicePosition);

      for (let a = 1; a < NX - 1 && vCount < maxV; a++) {
        // k は室内層(0〜NX-1)のみを対象とし、ノズル層(NX〜NZ-1)は除外する。
        // ノズル部はベクトルを出さず、ノズルメッシュ(スケルトン)のみで
        // 形状・存在を示す方針のため。
        for (let k = 1; k < NX - 1 && vCount < maxV; k++) {
          const c = yz ? this._idx(fIdx, a, k) : this._idx(a, fIdx, k);
          if (this.ptype[c] === this.WALL || this.ptype[c] === this.VOID) continue;
          if (this._writeVectorArrow(c, vCount, positions, colors)) vCount++;
        }
      }
    } else {
      // ── 3D表示: 全域をstride間引きでサンプリング(従来通り) ─────────
      const stride = 2;
      for (let i = 1; i < NX - 1 && vCount < maxV; i += stride) {
        for (let j = 1; j < NX - 1 && vCount < maxV; j += stride) {
          for (let k = 1; k < NX - 1 && vCount < maxV; k += stride) {
            const c = this._idx(i, j, k);
            if (this.ptype[c] === this.WALL || this.ptype[c] === this.VOID) continue;
            if (this._writeVectorArrow(c, vCount, positions, colors)) vCount++;
          }
        }
      }
    }

    for (let k = vCount; k < maxV; k++) {
      const baseP = k * 18;
      if (positions[baseP] === 999) break;
      for (let vert = 0; vert < 18; vert++) positions[baseP + vert] = 999;
    }

    this.vectorGeometry.attributes.position.needsUpdate = true;
    this.vectorGeometry.attributes.color.needsUpdate    = true;
  }

  updateView() {
    const signature = `${this.visualMode}:${this.sliceMode}:${this.nozzleFocus}`;
    if (signature !== this.viewSignature && this.camera) {
      if (this.visualMode !== 0) {
        // ── ノズル拡大 ──────────────────────────────────────────
        // 断面・コンター表示時のみ有効。注視点を室内中心(0,0,0)から
        // 天井のノズル軸中間高さ(z=(DOM+DOM_Z)/2 → ワールドy)へ移し、
        // カメラ距離を大きく縮めることでノズル形状とその近傍だけを
        // 拡大表示する。sliceMode(YZ/XZ)に応じた見る方向は維持する。
        const targetY = this.nozzleFocus ? this._toWorldY((this.DOM + this.DOM_Z) / 2) : 0;
        const dist    = this.nozzleFocus ? 0.7 : 4.5;
        const camY    = targetY + (this.nozzleFocus ? 0.12 : 0.2);
        if (this.sliceMode === 1) this.camera.position.set(dist, camY, 0);
        else this.camera.position.set(0, camY, dist);
        this.camera.lookAt(0, targetY, 0);
        this.controls?.target.set(0, targetY, 0);
      } else {
        this.camera.position.set(0, 0.8, 4.5);
        this.camera.lookAt(0, 0, 0);
      }
      this.viewSignature = signature;
    }
    this._updateVisuals();
  }

  // ────────────────────────────────────────────────────────────────
  //  ノズル部拡大トグル (断面・コンター表示時のみ意味を持つ)
  // ────────────────────────────────────────────────────────────────
  setNozzleFocus(on) {
    this.nozzleFocus = !!on;
    this.updateView();
  }

  tick() {
    if (!this.paused) {
      const t0 = performance.now();
      for (let s = 0; s < this.substepsPerFrame; s++) {
        this.step();
        // 1フレームあたり18msを超えたら即座に描画・UIスレッドへ処理を戻す
        // これによりCPUモードでもUIのフリーズや引っ掛かりを完全に防止し爆速レスポンスを維持
        if (performance.now() - t0 > 18) break;
      }
    }
    this._updateVisuals();
    if (this.controls) this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  async setNozzle(type) {
    this.paused = true;
    this.nozzleType = type;
    this.stepCount = 0;
    this._initParticles(type);
    this._initParticlePoints();
    this._updateSectionSurface();
    this._updateNozzleMesh();
    this.paused = false;
  }

  async reset() {
    this.paused = true;
    this.stepCount = 0;
    this._initParticles(this.nozzleType);
    this._initParticlePoints();
    this._updateSectionSurface();
  }

  setInletCondition(velocity, temp, pressure) {
    if (velocity !== undefined && !isNaN(velocity)) AC.inletVelocity = velocity;
    if (temp !== undefined && !isNaN(temp)) {
      AC.inletTemp = temp;
      this.T_IN = temp;
      this.stats.tMin = temp;
    }
    if (pressure !== undefined && !isNaN(pressure)) {
      this.inletPressure = pressure;
    }
    updateNozzleVelocities();
    this._applyInletCondition();
  }

  // ────────────────────────────────────────────────────────────────
  //  INLETセルの流入条件を再適用 (連続の式による局所速度 + 加圧圧力)
  // ────────────────────────────────────────────────────────────────
  _applyInletCondition() {
    const nozzle = NOZZLES[this.nozzleType];
    const core   = nozzle.innerCylinder || null;
    const pInlet = this.inletPressure ?? 60.0;
    for (let c = 0; c < this.N; c++) {
      if (this.ptype[c] !== this.INLET) continue;
      const zNoz = Math.min(this.NOZZLE_H, Math.max(0, this.pz[c] - this.DOM));
      const localV = this._nozzleLocalVelocity(this.nozzleType, nozzle, core, zNoz);
      this.vz[c] = -localV * (this.inletFrac ? this.inletFrac[c] : 1);
      this.temp[c] = this.T_IN;
      this.press[c] = pInlet;
    }
  }

  toggleTurbulence(on) {
    if (on !== undefined) this.useTurbulence = !!on;
    else this.useTurbulence = !this.useTurbulence;
  }

  _stepTurbulence() {
    const NX = this.NX, NZ = this.NZ, DT = this.DT;
    const xg = this.xg, yg = this.yg, zg = this.zg;
    const C_mu = this.C_mu, C_eps1 = this.C_eps1, C_eps2 = this.C_eps2;
    const sigma_k = this.sigma_k, sigma_eps = this.sigma_eps;
    const NU = this.NU;

    for (let i = 1; i < NX - 1; i++) {
    for (let j = 1; j < NX - 1; j++) {
    for (let k = 1; k < NZ - 1; k++) {
      const c = this._idx(i, j, k);
      if (this.ptype[c] !== this.FLUID) continue;

      const u = this.vx[c], v = this.vy[c], w = this.vz[c];
      const hx = xg[i+1] - xg[i-1], hy = yg[j+1] - yg[j-1], hz = zg[k+1] - zg[k-1];

      const dudx = (this._at(this.vx,i+1,j,k) - this._at(this.vx,i-1,j,k)) / hx;
      const dvdy = (this._at(this.vy,i,j+1,k) - this._at(this.vy,i,j-1,k)) / hy;
      const dwdz = (this._at(this.vz,i,j,k+1) - this._at(this.vz,i,j,k-1)) / hz;

      const dudy = (this._at(this.vx,i,j+1,k) - this._at(this.vx,i,j-1,k)) / hy;
      const dvdx = (this._at(this.vy,i+1,j,k) - this._at(this.vy,i-1,j,k)) / hx;
      const dudz = (this._at(this.vx,i,j,k+1) - this._at(this.vx,i,j,k-1)) / hz;
      const dwdx = (this._at(this.vz,i+1,j,k) - this._at(this.vz,i-1,j,k)) / hx;
      const dvdz = (this._at(this.vy,i,j,k+1) - this._at(this.vy,i,j,k-1)) / hz;
      const dwdy = (this._at(this.vz,i,j+1,k) - this._at(this.vz,i,j-1,k)) / hy;

      const S2 = 2*(dudx**2 + dvdy**2 + dwdz**2) + (dudy+dvdx)**2 + (dudz+dwdx)**2 + (dvdz+dwdy)**2;
      const nut_c = this.nu_t[c];
      const Pk = Math.min(10.0 * Math.max(1e-4, this.eps_turb[c]), nut_c * S2);

      // 風上移流
      const kc = this.k_turb[c];
      const advK_x = u >= 0 ? u * (kc - this._at(this.k_turb,i-1,j,k)) / (xg[i]-xg[i-1]) : u * (this._at(this.k_turb,i+1,j,k) - kc) / (xg[i+1]-xg[i]);
      const advK_y = v >= 0 ? v * (kc - this._at(this.k_turb,i,j-1,k)) / (yg[j]-yg[j-1]) : v * (this._at(this.k_turb,i,j+1,k) - kc) / (yg[j+1]-yg[j]);
      const advK_z = w >= 0 ? w * (kc - this._at(this.k_turb,i,j,k-1)) / (zg[k]-zg[k-1]) : w * (this._at(this.k_turb,i,j,k+1) - kc) / (zg[k+1]-zg[k]);

      const diffK_eff = NU + nut_c / sigma_k;
      const d2k = (this._at(this.k_turb,i+1,j,k) - 2*kc + this._at(this.k_turb,i-1,j,k)) / ((hx/2)**2) +
                  (this._at(this.k_turb,i,j+1,k) - 2*kc + this._at(this.k_turb,i,j-1,k)) / ((hy/2)**2) +
                  (this._at(this.k_turb,i,j,k+1) - 2*kc + this._at(this.k_turb,i,j,k-1)) / ((hz/2)**2);

      const eps_c = Math.max(1e-6, this.eps_turb[c]);
      const k_star = Math.max(1e-6, kc + DT * (-advK_x - advK_y - advK_z + diffK_eff * d2k + Pk));
      const k_new = k_star / (1.0 + DT * (eps_c / Math.max(1e-6, kc)));
      this.k_turb[c] = Math.max(1e-6, k_new);

      // eps 更新
      const advEps_x = u >= 0 ? u * (eps_c - this._at(this.eps_turb,i-1,j,k)) / (xg[i]-xg[i-1]) : u * (this._at(this.eps_turb,i+1,j,k) - eps_c) / (xg[i+1]-xg[i]);
      const advEps_y = v >= 0 ? v * (eps_c - this._at(this.eps_turb,i,j-1,k)) / (yg[j]-yg[j-1]) : v * (this._at(this.eps_turb,i,j+1,k) - eps_c) / (yg[j+1]-yg[j]);
      const advEps_z = w >= 0 ? w * (eps_c - this._at(this.eps_turb,i,j,k-1)) / (zg[k]-zg[k-1]) : w * (this._at(this.eps_turb,i,j,k+1) - eps_c) / (zg[k+1]-zg[k]);

      const diffEps_eff = NU + nut_c / sigma_eps;
      const d2eps = (this._at(this.eps_turb,i+1,j,k) - 2*eps_c + this._at(this.eps_turb,i-1,j,k)) / ((hx/2)**2) +
                    (this._at(this.eps_turb,i,j+1,k) - 2*eps_c + this._at(this.eps_turb,i,j-1,k)) / ((hy/2)**2) +
                    (this._at(this.eps_turb,i,j,k+1) - 2*eps_c + this._at(this.eps_turb,i,j,k-1)) / ((hz/2)**2);

      const eps_src = C_eps1 * (eps_c / Math.max(1e-6, kc)) * Pk;
      const eps_star = Math.max(1e-6, eps_c + DT * (-advEps_x - advEps_y - advEps_z + diffEps_eff * d2eps + eps_src));
      const eps_new = eps_star / (1.0 + DT * C_eps2 * (eps_c / Math.max(1e-6, kc)));
      this.eps_turb[c] = Math.max(1e-6, eps_new);

      this.nu_t[c] = Math.min(100.0 * NU, Math.max(0.0, C_mu * (this.k_turb[c]**2) / this.eps_turb[c]));
    }}}
  }

  destroy() {
    this.sectionSurfaceGeometry?.dispose();
    this.sectionGridGeometry?.dispose();
    this.vectorGeometry?.dispose();
    this.renderer?.dispose();
  }
}

// JavaScript版Jetカラーマップ
function jetColorJS(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if      (t < 0.125) { r=0;         g=0;              b=0.5+4*t; }
  else if (t < 0.375) { r=0;         g=4*(t-0.125);    b=1; }
  else if (t < 0.625) { r=4*(t-0.375); g=1;            b=1-4*(t-0.375); }
  else if (t < 0.875) { r=1;         g=1-4*(t-0.625);  b=0; }
  else                { r=1-4*(t-0.875); g=0;           b=0; }
  return [r, g, b];
}

if (typeof window !== 'undefined') {
  window.CPUFallbackMPS = CPUFallbackMPS;
  window.jetColorJS     = jetColorJS;
}