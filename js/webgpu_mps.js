'use strict';
/**
 * webgpu_mps.js
 * WebGPU MPS粒子法ソルバー + レンダラー
 *
 * 粒子配置:
 *   - 物理座標 [0,2]³ m
 *   - L0 = 0.067m → 31×31×31 ≈ 29,791 粒子
 *   - HASH_SIZE = 16384, MAX_CELL = 64
 *
 * シェーダーバインド:
 *   group(0): params(uniform), particles_A(storage), particles_B(storage),
 *             hashCount(atomic), hashData(storage), hashCountRead(storage)
 */

class LegacyWebGPUMPS {
  constructor() {
    // ── 物理定数 ──────────────────────────────────────────────────
    // L0 を室内・ノズル共通の 0.03m に統一する。
    //
    // 理由: 過去に L0r=0.05m(室内) + L0n=0.005m(ノズル) の二重解像度だった際、
    //   影響半径 RE=0.105m 内に ~700個のノズル粒子が入り
    //   n_i >> N0 → 圧力 = KAPPA*(700/14.4-1) ≈ KAPPA*47 となる。
    //   これは大気圧の数百倍に相当し計算不安定を招く。
    //   統一 L0=0.03m では密度が一様で MPS 圧力が正しく伝播する。
    //
    // ノズル Ø85mm / L0=0.035m ≈ 2.4粒子/直径 (粗いが最低限の流路表現)
    // 室内粒子数: 59×59×59 ≈ 205,000 (L0=0.030→0.035へ変更、-35%)
    this.L0        = 0.035;            // 統一粒子間隔 [m] (室内・ノズル共通) ※要望により0.030→0.035に変更 (粒子数 -35%)
    this.RE        = 2.1 * this.L0;   // 影響半径 [m] = 0.063m
    this.N0        = 14.4;            // 基準粒子数密度 (3D uniform)
    this.LAMBDA    = 1.703 * this.L0 * this.L0;  // ラプラシアン係数 [m²]
    this.DT        = 0.001;           // 時間刻み [s]
    this.NU        = 0.001;           // 乱流有効粘度 [m²/s] (空気物性 1.5e-5 の~67倍)
                                       // ※0.01(~700倍)は運動量拡散が強すぎ、噴流が室内奥まで
                                       //   届く前に減衰してしまう(「噴流が弱い」問題の一因)ため
                                       //   1/10に低減。乱流Prandtl数≈1を仮定しALPHAも同率で低減。
    this.ALPHA     = 0.001;           // 乱流有効熱拡散率 [m²/s] (NUと同率で低減、Pr_t≈1相当)
    this.G         = 9.81;            // 重力 [m/s²]
    this.BETA      = 1 / 307;         // 体積膨張係数 [1/K]
    this.T_REF     = 34.0;            // 周囲温度 [°C]
    this.T_IN      = 23.5;            // 流入温度 [°C]
    this.RHO0      = 1.2;             // 空気密度 [kg/m³]
    this.KAPPA     = 0.2;             // EOS剛性係数
    this.DOM       = 2.0;             // 室内領域サイズ [m]
    this.DOM_Z     = 2.18;            // z方向総高さ [m] = 室内2.0 + ノズル0.18
    this.NOZZLE_H  = 0.18;            // ノズル高さ [m]
    this.CELL_SIZE = this.RE;         // ハッシュセルサイズ

    // ── ハッシュ設定 ───────────────────────────────────────────────
    this.HASH_SIZE = 131072;  // 2^17
    // MAX_CELL: 32→192に増量。
    // 噴流コア柱・ノズル内部をREFINE_K=2.5倍の解像度で細分化するため、
    // CELL_SIZE(=RE)を変えずに同じハッシュ格子を使う場合、その領域の
    // セルには理論上 (RE/L0n)³ ≈ 145個/セル の粒子が入りうる
    // (質量重みmrで物理量自体は正しく保たれるが、ハッシュの格納容量は
    //  物理粒子数ベースなので別途確保が必要)。安全マージンを見て192に設定。
    // ※ より高い解像度(K=3, ~250個/セル)も可能だが、ハッシュバッファが
    //   131072×320×4byte≈167MBまで膨らみ低スペック環境で確保に失敗する
    //   リスクが上がるため、K=2.5・MAX_CELL=192を安全側の落とし所とした。
    this.MAX_CELL  = 192;     // セルあたり最大粒子数

    // ── 粒子数設定 ────────────────────────────────────────────────
    this.NX  = Math.ceil(this.DOM / this.L0) + 1;  // x,y方向 (2.0/0.035 = 57.1→58→59)
    this.NZ  = Math.ceil(this.DOM_Z / this.L0) + 1; // z方向
    this.N   = this.NX * this.NX * this.NZ;         // ≈ 59×59×63 ≈ 219k

    // ── WebGPU リソース ────────────────────────────────────────────
    this.device    = null;
    this.canvas    = null;
    this.context   = null;
    this.format    = 'bgra8unorm';

    // バッファ
    this.bufParA   = null;   // Particle buffer A (ping)
    this.bufParB   = null;   // Particle buffer B (pong)
    this.bufHash   = null;   // atomic hash count
    this.bufHashR  = null;   // non-atomic hash count (読み取り用)
    this.bufHashD  = null;   // hash data (粒子IDリスト)
    this.bufParams = null;   // uniform params
    this.bufCamera = null;   // uniform camera
    this.bufRoom   = null;   // room wireframe vertices
    this.bufRoomIdx= null;   // room wireframe indices
    this.bufSlice  = null;   // 断面補間値
    this.bufNozzle = null;   // ノズル3Dワイヤー
    this.depthTex  = null;   // depth texture

    // パイプライン
    this.pipeHashZero  = null;
    this.pipeHashBuild = null;
    this.pipeHashCopy  = null;
    this.pipeDensity   = null;
    this.pipeExplicit  = null;
    this.pipeTemp      = null;
    this.pipeInletBC   = null;
    this.pipeRender    = null;
    this.pipeRoom      = null;
    this.pipeSliceCompute = null;
    this.pipeSliceRender = null;
    this.pipeNozzle = null;

    // バインドグループ
    this.bgCompA   = null;   // A→B
    this.bgCompB   = null;   // B→A
    this.bgHash    = null;   // hash zero/build
    this.bgHashCp  = null;   // hash copy
    this.bgInlet   = null;   // inlet BC (A)
    this.bgInletB  = null;   // inlet BC (B)
    this.bgRenderA = null;   // render from A
    this.bgRenderB = null;   // render from B
    this.bgRoomBG  = null;   // room render
    this.bgSliceA  = null;
    this.bgSliceB  = null;
    this.bgSliceRender = null;
    this.bgNozzle = null;

    // 状態
    this.pingPong    = 0;    // 0: A→B, 1: B→A
    this.stepCount   = 0;
    this.nozzleType  = 1;
    this.displayMode = 1;    // 0=温度, 1=速度, 2=圧力
    this.visualMode  = 1;    // 0=3D粒子, 1=断面, 2=コンター
    this.sliceMode   = 1;    // 1=YZ面, 2=XZ面
    this.slicePosition = 1.0;
    this.viewSignature = '';
    this.paused      = true;

    // カメラ
    this.camera = {
      theta  : Math.PI / 2,    // rad 水平回転 (YZ面正面)
      phi    : Math.PI / 2 - 0.04, // rad 垂直角
      radius : 4.0,    // 距離
      target : [0, 0, 0],  // 注視点
    };
    this._drag = null;

    // 統計
    this.stats = { tMin: 23.5, tMax: 34, vMax: 0, step: 0, time: 0 };
    this.substepsPerFrame = 5;
    this.steadyMode = true;
    this.steadySteps = 2000;
    this.steadyComplete = false;
  }

  // ────────────────────────────────────────────────────────────────
  //  初期化
  // ────────────────────────────────────────────────────────────────
  async init(canvas, nozzleType = 1) {
    this.canvas = canvas;
    this.nozzleType = nozzleType;

    // ── WebGPU デバイス取得 ──────────────────────────────────────
    if (!navigator.gpu) throw new Error('WebGPU未対応ブラウザです');
      const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU アダプター取得失敗');
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    this.device.lost.then(info => {
      console.error('WebGPU device lost:', info.message);
    });

    // ── Canvas設定 ───────────────────────────────────────────────
    // 初期化時にcanvasサイズを確定させる
    if (!canvas.width || !canvas.height) {
      canvas.width  = canvas.clientWidth  || 800;
      canvas.height = canvas.clientHeight || 600;
    }
    this.format  = navigator.gpu.getPreferredCanvasFormat();
    this.context = canvas.getContext('webgpu');
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied',
    });

    // ── バッファ作成 ─────────────────────────────────────────────
    await this._createBuffers(nozzleType);

    // ── パイプライン作成 ──────────────────────────────────────────
    await this._createPipelines();

    // ── バインドグループ作成 ──────────────────────────────────────
    this._createBindGroups();

    // ── カメライベント設定 ────────────────────────────────────────
    this._setupCamera();

    return this;
  }

  // ────────────────────────────────────────────────────────────────
  //  粒子バッファ初期化
  // ────────────────────────────────────────────────────────────────
  async _createBuffers(nozzleType) {
    const dev    = this.device;
    const nozzle = NOZZLES[nozzleType];
    const DOM    = this.DOM;
    const DOM_Z  = this.DOM_Z;
    const NZ_H   = this.NOZZLE_H;
    const L0r    = this.L0;           // 室内粒子間隔 [m] (粗)
    // ── 質量スケーリング型 多重解像度MPS ──────────────────────────────
    // 室内粒子間隔 L0r をそのまま使うとノズル出口(Ø85mm)が~2.4粒子/直径
    // にしかならず、噴流の断面を粒子法で安定に表現できない(過去の実測で
    // 噴流が塊状に崩壊する主因)。かといって単純に間隔だけ細かくすると
    // (旧: L0r=0.05/L0n=0.005の二重解像度)、質量重みなしではRE内粒子数が
    // ~700に達し n_i/N0が爆発して圧力が発散した。
    // → 今回は「質量比(体積比)mr」を粒子ごとに持たせ、数密度・圧力勾配・
    //   粘性のカーネル総和を mr で重み付けする(shaders.js の mpsWm)。
    //   これによりRE・N0はグローバルのまま、局所的に粒子を細かくしても
    //   物理量が正しいスケールに保たれる。
    // REFINE_K: セル内最大粒子数 (RE/L0n)³ とハッシュ容量(MAX_CELL)の
    // トレードオフで2.5に設定。出口Ø85mm/L0n≈6.1粒子/直径を確保しつつ、
    // 局所セル最大粒子数を~145に抑えMAX_CELL=192で安全に収まる。
    const REFINE_K = 2.5;                     // 細分化係数
    const L0n      = this.L0 / REFINE_K;      // 微細粒子間隔 [m] ≈ 0.0140m
    const MR_FINE  = 1.0 / (REFINE_K ** 3);   // 微細粒子の質量比 ≈ 1/15.6
    const NX     = this.NX;
    const cx     = DOM / 2, cy = DOM / 2;  // 室中心 (1.0m, 1.0m)

    const STRIDE = 16;  // float32 per particle (pos4 + vel4 + aux4 + aux2:mr等4)
    // 動的配列でパーティクルを収集 (ノズル円柱配置で粒子数が可変)
    const raw = [];

    // nozzleTag: aux.w に格納するノズル粒子フラグ
    //   1.0 = ノズル内部粒子 (MPS自由流体だが室内流出時に再投入する)
    //   0.0 = 室内粒子 (通常のMPS流体)
    // ※ aux.w は SHADER_DENSITY/SHADER_EXPLICIT/SHADER_HEAT_DIFFUSE で
    //   書き込まれないため、フラグは全ステップ永続する。
    // mr: aux2.x に格納する質量比(体積比)。室内粒子=1.0、微細化領域=MR_FINE。
    const pushPart = (x, y, z, ptype, vz, temp, nozzleTag = 0, mr = 1.0) => {
      const wt = (ptype === 3) ? 1 : ptype;
      raw.push(
        x, y, z, wt,                          // pos.xyzw
        0, 0, vz, 0,                          // vel (vx=vy=0, pressure=0)
        temp, this.N0, Math.abs(vz), nozzleTag,// aux (temp, ni, speed, nozzleTag)
        mr, 0, 0, 0                           // aux2 (質量比mr, 予約×3)
      );
    };

    // 噴流コアバッファ柱のジオメトリ (ノズル出口直下を高解像度化する円柱領域)
    const { rOuter: outletROuter } = getNozzleSection(nozzleType, 0);
    const bufferRadius = outletROuter * 1.3;  // 柱半径 (出口外径の1.3倍、剪断層まで含む)
    const bufferDepth  = 0.40;                 // 天井からの深さ [m] (SHADER_EXPLICITのjet_depthと一致させる)

    // ════════════════════════════════════════════════════════════════
    // 1. 室内領域 (z: 0 → DOM): 均一 Cartesian 格子 L0r
    //
    //    【重要】MPS法では N0=14.4 が均一格子 L0=0.035m の前提で計算されている。
    //    tanh 非均一配置を使うと、粒子間隔が場所ごとに変わるため初期から
    //    n_i ≠ N0 となり「n_i > N0 → 反発圧力」「n_i < N0 → 負圧」の
    //    人工的な圧力場が発生して数値不安定の原因となる。
    //    → MPS 室内粒子は均一格子のまま維持する。
    //
    //    ノズル直下の解像度向上は Section 2 のバッファ柱（L0n 微細粒子＋
    //    質量比 mr）で対応する。bufferRadius と bufferDepth を大きくすることで
    //    ノズル直下の広範囲を高解像度で充填できる（N0 問題が起きない理由:
    //    微細粒子は mr=1/REFINE_K³ で重み付けされるため n_i の総和は均一
    //    格子とほぼ等価に保たれる）。
    //    NX=59 → 59×59×59 ≈ 205k粒子 (噴流コアバッファ柱の範囲は生成しない)
    // ════════════════════════════════════════════════════════════════
    const NZ_room = Math.ceil(DOM / L0r) + 1;
    for (let iz = 0; iz < NZ_room; iz++) {
      const z = Math.min(iz * L0r, DOM);
      for (let iy = 0; iy < NX; iy++) {
        for (let ix = 0; ix < NX; ix++) {
          const x   = ix * L0r;
          const y   = iy * L0r;
          const dx  = x - cx, dy = y - cy;
          const r2d = Math.sqrt(dx*dx + dy*dy);

          const isFloor    = (iz === 0);
          const isCeil     = (Math.abs(z - DOM) < 1e-9);
          const isSideWall = (ix === 0 || ix === NX-1 || iy === 0 || iy === NX-1);
          // 噴流コアバッファ柱の範囲内かどうか (天井直下bufferDepth かつ柱半径内)
          const inBufferCol = (z >= DOM - bufferDepth) && (r2d < bufferRadius);

          let ptype = 1, vz = 0, temp = this.T_REF;

          if (isFloor || isSideWall) {
            ptype = 1;  // 床・側壁
          } else if (isCeil) {
            // 天井開口部分は下の「噴流コアバッファ柱」セクションで
            // 高解像度(L0n)の開口リングとして生成するため、ここではスキップ。
            const isCoreBlocked = nozzle.outlet.hasCore && r2d < nozzle.outlet.coreR;
            if (r2d <= outletROuter + L0n && !isCoreBlocked) {
              continue;
            }
            ptype = 1;  // 天井壁面 (開口外)
          } else if (inBufferCol) {
            // 噴流コアバッファ柱セクションで高解像度粒子として生成するためスキップ
            continue;
          } else {
            ptype = 0;  // 室内流体 (通常解像度)
            vz    = (Math.random() - 0.5) * 0.005;
          }
          pushPart(x, y, z, ptype, vz, temp);
        }
      }
    }


    // ════════════════════════════════════════════════════════════════
    // 2. 噴流コアバッファ柱 (z: DOM-bufferDepth → DOM): 円柱座標 L0n(微細)
    //    ノズルと同じ細かい解像度で天井直下を満たし、噴流入口BCゾーン
    //    (SHADER_EXPLICITのjet_depth)を抜けた直後に粒子数不足で噴流が
    //    崩壊するのを防ぐ。半径は出口外径の1.3倍(剪断層まで含む)。
    // ════════════════════════════════════════════════════════════════
    const nBufZ = Math.max(1, Math.round(bufferDepth / L0n));
    for (let iz_buf = 0; iz_buf <= nBufZ; iz_buf++) {
      const z = DOM - bufferDepth + iz_buf * (bufferDepth / nBufZ);
      const isCeilPlane = (iz_buf === nBufZ);  // z=DOM: 天井開口面 (旧ロジックのceil開口を継承)
      const depthFrac   = 1.0 - iz_buf / nBufZ;  // 0=天井, 1=柱下端

      for (let ring = 0; ring * L0n <= bufferRadius + L0n * 0.5; ring++) {
        const r    = ring * L0n;
        const nAng = (ring === 0) ? 1 : Math.max(6, Math.round(2 * Math.PI * r / L0n));

        for (let k = 0; k < nAng; k++) {
          const theta = (ring === 0) ? 0 : (2 * Math.PI * k / nAng);
          const px = cx + r * Math.cos(theta);
          const py = cy + r * Math.sin(theta);
          if (px < 0 || px > DOM || py < 0 || py > DOM) continue;

          // ノズル2の内部シリンダー直下は空洞 (中心低速域)
          const isCoreBlocked = nozzle.outlet.hasCore && r < nozzle.outlet.coreR;
          if (isCoreBlocked) continue;

          // 出口速度(連続の式で確定済み: nozzle.outlet.velocity)を基準に、
          // 柱下端に向けて緩やかに減衰させ周囲の室内粒子と滑らかに接続する
          const vTop = -nozzle.outlet.velocity;
          const vz   = vTop * (1.0 - 0.4 * depthFrac);
          const temp = this.T_IN + (this.T_REF - this.T_IN) * 0.15 * depthFrac;

          // isCeilPlaneの環は「天井開口リング」を兼ねるためnozzleTag=0、
          // それ以外は室内に流出後ノズル上面へ再投入する対象としてtag=1
          pushPart(px, py, z, 0, vz, temp, isCeilPlane ? 0 : 1, MR_FINE);
        }
      }
    }

    // ════════════════════════════════════════════════════════════════
    // 3. ノズル領域 (z: DOM → DOM_Z): 円柱座標 L0n(微細)
    //    ノズル Ø85mm / L0n≈0.0140m ≈ 6.1粒子/直径
    //    (旧: 室内と統一L0=0.035mで2.4粒子/直径だったが、質量重み付き
    //     多重解像度MPSにより圧力発散を起こさず解像度のみ向上できる)
    //    各高さレベルで同心円リスト状に配置 (ring=0が中心粒子)
    // ════════════════════════════════════════════════════════════════
    const nNozZ = Math.ceil(NZ_H / L0n);
    for (let iz_noz = 1; iz_noz <= nNozZ; iz_noz++) {
      const z    = Math.min(DOM + iz_noz * L0n, DOM_Z);
      const zNoz = Math.min(z - DOM, NZ_H);
      const { rInner, rOuter } = getNozzleSection(nozzleType, zNoz);
      const isTopBC = (iz_noz >= nNozZ);  // ノズル最上部 = 流入BC面

      // 連続の式による局所速度 (断面積保存則)
      const coreR = (nozzle.outlet.hasCore && nozzle.innerCylinder &&
                     zNoz >= nozzle.innerCylinder.zStart)
                    ? nozzle.innerCylinder.rOut : 0;
      const flowArea   = Math.max(Math.PI * (rInner*rInner - coreR*coreR), 1e-8);
      const localV     = Math.min(AC.inletVelocity * AC.inletArea / flowArea, 20);

      // 同心円リング配置: ring=0が中心(r=0), ring=1がr=L0n, ...
      for (let ring = 0; ring * L0n <= rOuter + L0n * 0.5; ring++) {
        const r    = ring * L0n;
        const nAng = (ring === 0) ? 1 : Math.max(6, Math.round(2 * Math.PI * r / L0n));

        for (let k = 0; k < nAng; k++) {
          const theta = (ring === 0) ? 0 : (2 * Math.PI * k / nAng);
          const px = cx + r * Math.cos(theta);
          const py = cy + r * Math.sin(theta);

          // ドメイン外スキップ
          if (px < 0 || px > DOM || py < 0 || py > DOM) continue;

          // ノズル2 内部シリンダー判定
          const isCoreBlocked = nozzle.outlet.hasCore &&
            nozzle.innerCylinder && zNoz >= nozzle.innerCylinder.zStart &&
            r < nozzle.innerCylinder.rOut;

          let ptype = 3, vz = 0, temp = this.T_IN;

          if (r < rInner && !isCoreBlocked) {
            // ── ノズル内流体: 流動物理をMPSに任せる設計 ──────────────
            // ・ノズル上面(isTopBC)のみ ptype=2 (流入BC: 速度固定)
            // ・内部は ptype=0 (自由MPS流体) で自然な速度場を形成
            //   - SHADER_DENSITY が正確な数密度を計算
            //   - SHADER_EXPLICIT が圧力勾配・粘性・重力を適用
            //   - ノズル壁 (ptype=1) が流路形状を規定
            // ・aux.w=1.0 (ノズルフラグ) で流出時の再投入を識別
            //
            // ★ Bug fix: vz=-localV を continue より前に移動する。
            //   以前は continue の後に置かれており、内部粒子が常に vz=0 で
            //   初期化されていた（デッドコード）。これにより噴流が計算開始直後
            //   から正しい速度を持つよう修正。
            vz = -localV;  // 連続の式による局所軸方向速度 (下向き)
            if (isTopBC) {
              const coreBlkOut = nozzle.outlet.hasCore && r < nozzle.outlet.coreR;
              ptype = coreBlkOut ? 1 : 2;  // 上面: 流入BC
              pushPart(px, py, z, ptype, 0, temp, 0, MR_FINE);  // BC粒子: 速度はSHADER_INLET_BCで設定
            } else {
              ptype = 0;  // 内部: 自由MPS流体
              pushPart(px, py, z, ptype, vz, temp, 1.0, MR_FINE);  // nozzleTag=1.0, 初速あり
            }
            continue;  // 以下の共通pushをスキップ
          } else if (r <= rOuter) {
            ptype = 1;  // ノズル壁面 (no-slip)
            temp  = this.T_REF;
            vz    = 0;
          } else {
            continue;  // ノズル外は追加しない
          }

          if (ptype !== 3) pushPart(px, py, z, ptype, vz, temp, 0, MR_FINE);
        }
      }
    }

    // ── バッファ確保 ────────────────────────────────────────────────
    this.N = raw.length / STRIDE;
    const data    = new Float32Array(raw);
    const bufSize = data.byteLength;

    // Particle Buffer A
    this.bufParA = dev.createBuffer({
      label: 'ParticleA',
      size : bufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    dev.queue.writeBuffer(this.bufParA, 0, data);

    // Particle Buffer B (初期値コピー)
    this.bufParB = dev.createBuffer({
      label: 'ParticleB',
      size : bufSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    dev.queue.writeBuffer(this.bufParB, 0, data);

    // Hash Count Buffer (atomic u32)
    this.bufHash = dev.createBuffer({
      label: 'HashCount',
      size : this.HASH_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Hash Count Read Buffer (non-atomic)
    this.bufHashR = dev.createBuffer({
      label: 'HashCountRead',
      size : this.HASH_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Hash Data Buffer (粒子IDリスト)
    this.bufHashD = dev.createBuffer({
      label: 'HashData',
      size : this.HASH_SIZE * this.MAX_CELL * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Params Uniform Buffer (128 bytes)
    this.bufParams = dev.createBuffer({
      label: 'Params',
      size : 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._updateParamsBuffer();

    // Camera Uniform Buffer (80 bytes → 96 aligned)
    this.bufCamera = dev.createBuffer({
      label: 'Camera',
      size : 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._updateCameraBuffer();

    // ── 粒子ソート用バッファ (カメラ距離バイトニックソート) ──────────
    // 半透明ビルボードを正しい重なり順(遠→近)で描画するため、描画直前に
    // 毎フレーム並べ替える。配列長は2の冪に切り上げる必要がある。
    this.sortPaddedN = 256;
    while (this.sortPaddedN < this.N) this.sortPaddedN <<= 1;

    this.bufSortKey = dev.createBuffer({
      label: 'SortKey',
      size : this.sortPaddedN * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.bufSortIdx = dev.createBuffer({
      label: 'SortIdx',
      size : this.sortPaddedN * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.bufSortParams = dev.createBuffer({
      label: 'SortParams',
      size : 16,   // n, paddedN, k, j (u32 × 4)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._sortParamsCPU = new Uint32Array(4);

    // バイトニックソートの(k,j)ステージ列を事前計算 (フレーム間で不変)
    this._sortStages = [];
    for (let k = 2; k <= this.sortPaddedN; k <<= 1) {
      for (let j = k >> 1; j > 0; j >>= 1) {
        this._sortStages.push([k, j]);
      }
    }

    // Depth Texture
    this._createDepthTexture();

    // Room Wireframe Buffer
    this._createRoomBuffers();
    this._createNozzleBuffers(nozzleType);
  }

  // ────────────────────────────────────────────────────────────────
  //  Params ユニフォームバッファ更新
  // ────────────────────────────────────────────────────────────────
  _updateParamsBuffer() {
    const n   = NOZZLES[this.nozzleType];
    const buf = new ArrayBuffer(128);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);

    // u32 fields
    u32[0] = this.N;           // N
    u32[1] = this.HASH_SIZE;   // HASH_SIZE
    u32[2] = this.MAX_CELL;    // MAX_CELL
    u32[3] = this.stepCount;   // step

    // f32 fields (starting at byte offset 16 = index 4)
    f32[4] = this.L0;          // L0
    f32[5] = this.RE;          // RE
    f32[6] = this.RE * this.RE;// RE2
    f32[7] = this.N0;          // N0

    f32[8]  = this.LAMBDA;     // LAMBDA
    f32[9]  = this.DT;         // DT
    f32[10] = this.NU;         // NU
    f32[11] = this.ALPHA;      // ALPHA

    f32[12] = this.G;          // G
    f32[13] = this.BETA;       // BETA
    f32[14] = this.T_REF;      // T_REF
    f32[15] = this.T_IN;       // T_IN

    f32[16] = this.RHO0;       // RHO0
    f32[17] = this.KAPPA;      // KAPPA
    f32[18] = this.CELL_SIZE;  // CELL_SIZE
    f32[19] = this.DOM;        // DOM (室内: 2.0m)

    f32[20] = AC.inletVelocity; // V_IN: ノズル上面の流入速度
    u32[21] = this.nozzleType;   // NOZZLE
    f32[22] = n.outlet.innerR;   // IN_R (ノズル出口半径)
    f32[23] = this.DOM / 2;      // IN_CX

    f32[24] = this.DOM / 2;      // IN_CY
    u32[25] = n.outlet.hasCore ? 1 : 0;  // HAS_CORE
    f32[26] = n.outlet.coreR;    // CORE_R
    f32[27] = this.T_IN;         // T_MIN

    f32[28] = this.T_REF;        // T_MAX
    f32[29] = n.outlet.velocity * 1.5; // V_MAX
    u32[30] = this.displayMode;  // DISP_MODE
    f32[31] = this.DOM_Z;        // DOM_Z (z方向総高さ: 2.18m)

    this.device.queue.writeBuffer(this.bufParams, 0, buf);
  }

  // ────────────────────────────────────────────────────────────────
  //  カメラ行列計算 & バッファ更新
  // ────────────────────────────────────────────────────────────────
  _updateCameraBuffer() {
    const { theta, phi, radius, target } = this.camera;
    const sin_phi   = Math.sin(phi);
    const cos_phi   = Math.cos(phi);
    const sin_theta = Math.sin(theta);
    const cos_theta = Math.cos(theta);

    // カメラ位置 (球座標)
    const ex = target[0] + radius * sin_phi * sin_theta;
    const ey = target[1] + radius * cos_phi;
    const ez = target[2] + radius * sin_phi * cos_theta;

    // View行列 (lookAt)
    const mvp = this._makeViewProjection(
      [ex, ey, ez], target, [0, 1, 0]
    );

    const buf = new ArrayBuffer(96);
    const f32 = new Float32Array(buf);
    // mat4x4f = 16 floats
    for (let i = 0; i < 16; i++) f32[i] = mvp[i];
    // viewPos (padding to 96 bytes)
    f32[16] = ex; f32[17] = ey; f32[18] = ez; f32[19] = 1;
    f32[20] = this.visualMode === 0 ? 0 : this.sliceMode;
    f32[21] = this.slicePosition;
    f32[22] = this.visualMode === 2 ? this.L0 * 2.5 : this.L0 * 0.8;
    f32[23] = this.visualMode;

    this.device.queue.writeBuffer(this.bufCamera, 0, buf);
  }

  _makeViewProjection(eye, target, up) {
    const aspect = (this.canvas.width || 800) / (this.canvas.height || 600);
    const fov  = 60 * Math.PI / 180;
    const near = 0.1, far = 100;
    const f    = 1 / Math.tan(fov / 2);

    // ── 視点基底ベクトル ──────────────────────────────────────────
    let fwx = target[0]-eye[0], fwy = target[1]-eye[1], fwz = target[2]-eye[2];
    const fl = Math.sqrt(fwx*fwx + fwy*fwy + fwz*fwz);
    fwx/=fl; fwy/=fl; fwz/=fl;  // forward (eye→target)

    // right = forward × up
    let rx = fwy*up[2]-fwz*up[1], ry = fwz*up[0]-fwx*up[2], rz = fwx*up[1]-fwy*up[0];
    const rl = Math.sqrt(rx*rx+ry*ry+rz*rz);
    rx/=rl; ry/=rl; rz/=rl;

    // actual up = right × forward
    const ux = ry*fwz-rz*fwy, uy = rz*fwx-rx*fwz, uz = rx*fwy-ry*fwx;

    // ── View行列 (column-major, right-handed: -z=前方) ────────────
    // back = -forward  →  tz = -dot(back, eye) = +dot(forward, eye)
    const tx = -(rx*eye[0] + ry*eye[1] + rz*eye[2]);
    const ty = -(ux*eye[0] + uy*eye[1] + uz*eye[2]);
    const tz =  (fwx*eye[0]+ fwy*eye[1]+ fwz*eye[2]); // ← 符号 + が正しい

    // column-major: col0,col1,col2,col3
    const V = [
      rx,  ux, -fwx, 0,
      ry,  uy, -fwy, 0,
      rz,  uz, -fwz, 0,
      tx,  ty,  tz,  1,
    ];

    // ── Projection行列 (column-major, WebGPU NDC z∈[0,1]) ────────
    const P = [
      f/aspect, 0,  0,                    0,
      0,        f,  0,                    0,
      0,        0,  far/(near-far),       -1,
      0,        0,  near*far/(near-far),   0,
    ];

    // M = P × V (column-major行列積)
    const M = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += P[k*4+row] * V[col*4+k];
        M[col*4+row] = s;
      }
    }
    return M;
  }

  // ────────────────────────────────────────────────────────────────
  //  深度テクスチャ作成
  // ────────────────────────────────────────────────────────────────
  _createDepthTexture() {
    const w = Math.max(1, this.canvas.width  || this.canvas.clientWidth  || 800);
    const h = Math.max(1, this.canvas.height || this.canvas.clientHeight || 600);
    if (this.depthTex) this.depthTex.destroy();
    this.depthTex = this.device.createTexture({
      size  : [w, h],
      format: 'depth24plus',
      usage : GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  // ────────────────────────────────────────────────────────────────
  //  部屋ワイヤーフレームバッファ
  // ────────────────────────────────────────────────────────────────
  _createRoomBuffers() {
    const d = 1.0;  // 半辺長 (Three.js座標: [-1,1]³)
    // 8頂点
    const verts = new Float32Array([
      -d,-d,-d,  d,-d,-d,  d,d,-d, -d,d,-d,
      -d,-d, d,  d,-d, d,  d,d, d, -d,d, d,
    ]);
    // 12辺 × 2頂点 = 24インデックス
    const idx = new Uint16Array([
      0,1, 1,2, 2,3, 3,0,  // 底面
      4,5, 5,6, 6,7, 7,4,  // 上面
      0,4, 1,5, 2,6, 3,7,  // 縦辺
    ]);

    this.bufRoom = this.device.createBuffer({
      size : verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.bufRoom, 0, verts);

    this.bufRoomIdx = this.device.createBuffer({
      size : idx.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.bufRoomIdx, 0, idx);
    this.bufSlice = this.device.createBuffer({
      label: 'SliceValues',
      size: 65 * 65 * 4,   // 65x65グリッド (64x64セル)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  _createNozzleBuffers(nozzleType) {
    this.bufNozzle?.destroy();
    const vertices = [];
    const rings = 10;
    const segments = 24;
    const center = this.DOM / 2;
    const addLine = (a, b) => vertices.push(...a, ...b);
    for (let iz = 0; iz < rings; iz++) {
      const z = this.NOZZLE_H * iz / (rings - 1);
      const { rInner, rOuter } = getNozzleSection(nozzleType, z);
      for (let ring = 0; ring < 2; ring++) {
        const radius = ring === 0 ? rOuter : rInner;
        for (let s = 0; s < segments; s++) {
          const a0 = Math.PI * 2 * s / segments;
          const a1 = Math.PI * 2 * (s + 1) / segments;
          addLine(
            [radius * Math.cos(a0), z, radius * Math.sin(a0)],
            [radius * Math.cos(a1), z, radius * Math.sin(a1)],
          );
        }
      }
      if (iz > 0) {
        const prevZ = this.NOZZLE_H * (iz - 1) / (rings - 1);
        const prev = getNozzleSection(nozzleType, prevZ);
        for (let s = 0; s < segments; s++) {
          const a = Math.PI * 2 * s / segments;
          for (const [r0, r1] of [[prev.rOuter, rOuter], [prev.rInner, rInner]]) {
            addLine(
              [r0 * Math.cos(a), prevZ, r0 * Math.sin(a)],
              [r1 * Math.cos(a), z, r1 * Math.sin(a)],
            );
          }
        }
      }
    }
    const data = new Float32Array(vertices.length);
    for (let i = 0; i < vertices.length; i += 3) {
      data[i] = vertices[i];
      data[i + 1] = vertices[i + 1] + this.DOM - 1.0;
      data[i + 2] = vertices[i + 2];
    }
    this.nozzleVertexCount = data.length / 3;
    this.bufNozzle = this.device.createBuffer({
      label: 'NozzleWireframe', size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.bufNozzle, 0, data);
  }

  // ────────────────────────────────────────────────────────────────
  //  パイプライン作成
  // ────────────────────────────────────────────────────────────────
  async _createPipelines() {
    const dev = this.device;

    // ── コンピュートパイプライン共通ヘルパー ───────────────────────
    const mkComp = (label, code, entry = 'main') =>
      dev.createComputePipeline({
        label,
        layout : 'auto',
        compute: { module: dev.createShaderModule({ code }), entryPoint: entry },
      });

    this.pipeHashZero  = mkComp('HashZero',  SHADER_HASH_ZERO);
    this.pipeHashBuild = mkComp('HashBuild', SHADER_HASH_BUILD);
    this.pipeHashCopy  = mkComp('HashCopy',  SHADER_HASH_COPY);
    this.pipeDensity   = mkComp('Density',   SHADER_DENSITY);
    this.pipeExplicit  = mkComp('Explicit',  SHADER_EXPLICIT);
    this.pipeTemp      = mkComp('Temp',      SHADER_TEMPERATURE);
    this.pipeInletBC   = mkComp('InletBC',   SHADER_INLET_BC);
    this.pipeSliceCompute = mkComp('SliceCompute', SHADER_SLICE_COMPUTE);
    this.pipeSortInit  = mkComp('SortInit', SHADER_SORT_INIT);
    this.pipeSortStep  = mkComp('SortStep', SHADER_SORT_STEP);

    // ── レンダーパイプライン (粒子) ────────────────────────────────
    const vertMod = dev.createShaderModule({ code: SHADER_VERT });
    const fragMod = dev.createShaderModule({ code: SHADER_FRAG });

      this.pipeRender = dev.createRenderPipeline({
      label : 'ParticleRender',
      layout: 'auto',
      vertex: {
        module    : vertMod,
        entryPoint: 'vs_main',
      },
      fragment: {
        module    : fragMod,
        entryPoint: 'fs_main',
        targets   : [{
          format : this.format,
          blend  : {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive   : { topology: 'triangle-strip' },
        depthStencil: { format: 'depth24plus' },
    });

    // ── レンダーパイプライン (部屋ワイヤーフレーム) ────────────────
    const roomVertMod = dev.createShaderModule({ code: SHADER_ROOM_VERT });
    const roomFragMod = dev.createShaderModule({ code: SHADER_ROOM_FRAG });

    this.pipeRoom = dev.createRenderPipeline({
      label : 'RoomRender',
      layout: 'auto',
      vertex: {
        module    : roomVertMod,
        entryPoint: 'vs_room',
        buffers   : [{
          arrayStride: 12,  // 3 × float32
          attributes : [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        }],
      },
      fragment: {
        module    : roomFragMod,
        entryPoint: 'fs_room',
        targets   : [{
          format: this.format,
          blend : {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive   : { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
    });

    const sliceVert = dev.createShaderModule({ code: SHADER_SLICE_VERT });
    const sliceFrag = dev.createShaderModule({ code: SHADER_SLICE_FRAG });
    this.pipeSliceRender = dev.createRenderPipeline({
      label: 'SliceRender',
      layout: 'auto',
      vertex: { module: sliceVert, entryPoint: 'main' },
      fragment: {
        module: sliceFrag,
        entryPoint: 'main',
        targets: [{ format: this.format, blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        }}],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
    });

    const nozzleVert = dev.createShaderModule({ code: SHADER_NOZZLE_VERT });
    const nozzleFrag = dev.createShaderModule({ code: SHADER_NOZZLE_FRAG });
    this.pipeNozzle = dev.createRenderPipeline({
      label: 'NozzleRender', layout: 'auto',
      vertex: {
        module: nozzleVert, entryPoint: 'main',
        buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
      },
      fragment: { module: nozzleFrag, entryPoint: 'main', targets: [{
        format: this.format,
        blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } },
      }] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
    });
  }

  // ────────────────────────────────────────────────────────────────
  //  バインドグループ作成
  // ────────────────────────────────────────────────────────────────
  _createBindGroups() {
    const dev = this.device;
    const mkBG = (pipe, label, entries) =>
      dev.createBindGroup({ label, layout: pipe.getBindGroupLayout(0), entries });

    const entry = (binding, resource) => ({ binding, resource });
    const buf   = (b, offset = 0, size) => ({ buffer: b, offset, size });

    // ── ハッシュゼロ ───────────────────────────────────────────────
    this.bgHashZeroA = mkBG(this.pipeHashZero, 'HashZeroA', [
      entry(0, buf(this.bufParams)),
      entry(1, buf(this.bufHash)),
    ]);

    // ── ハッシュビルド (両バッファ) ────────────────────────────────
    this.bgHashBuildA = mkBG(this.pipeHashBuild, 'HashBuildA', [
      entry(0, buf(this.bufParams)),
      entry(1, buf(this.bufParA)),
      entry(2, buf(this.bufHash)),
      entry(3, buf(this.bufHashD)),
    ]);
    this.bgHashBuildB = mkBG(this.pipeHashBuild, 'HashBuildB', [
      entry(0, buf(this.bufParams)),
      entry(1, buf(this.bufParB)),
      entry(2, buf(this.bufHash)),
      entry(3, buf(this.bufHashD)),
    ]);

    // ── ハッシュコピー ────────────────────────────────────────────
    this.bgHashCp = mkBG(this.pipeHashCopy, 'HashCopy', [
      entry(0, buf(this.bufParams)),
      entry(1, buf(this.bufHash)),
      entry(2, buf(this.bufHashR)),
    ]);

    // ── コンピュートパス: A→B ─────────────────────────────────────
    const compEntries = (inBuf, outBuf) => [
      entry(0, buf(this.bufParams)),
      entry(1, buf(inBuf)),
      entry(2, buf(outBuf)),
      entry(3, buf(this.bufHashR)),
      entry(4, buf(this.bufHashD)),
    ];
    this.bgDensityA   = mkBG(this.pipeDensity,  'DensityA',  compEntries(this.bufParA, this.bufParB));
    this.bgDensityB   = mkBG(this.pipeDensity,  'DensityB',  compEntries(this.bufParB, this.bufParA));
    this.bgExplicitA  = mkBG(this.pipeExplicit, 'ExplicitA', compEntries(this.bufParA, this.bufParB));
    this.bgExplicitB  = mkBG(this.pipeExplicit, 'ExplicitB', compEntries(this.bufParB, this.bufParA));
    this.bgTempA      = mkBG(this.pipeTemp,     'TempA',     compEntries(this.bufParA, this.bufParB));
    this.bgTempB      = mkBG(this.pipeTemp,     'TempB',     compEntries(this.bufParB, this.bufParA));

    // ── 流入BC ────────────────────────────────────────────────────
    const inletEntries = (buf_) => [
      entry(0, buf(this.bufParams)),
      entry(1, buf(buf_)),
    ];
    this.bgInletA = mkBG(this.pipeInletBC, 'InletA', inletEntries(this.bufParA));
    this.bgInletB = mkBG(this.pipeInletBC, 'InletB', inletEntries(this.bufParB));

    // ── レンダリング ──────────────────────────────────────────────
    // sortIdxは(A/Bどちらの粒子バッファを描くかに関わらず)常に同じ
    // 1本のバッファ - 描画直前に_sortParticles()でその都度書き換える。
    const renderEntries = (par) => [
      entry(0, buf(this.bufParams)),
      entry(1, buf(par)),
      entry(2, buf(this.bufCamera)),
      entry(3, buf(this.bufSortIdx)),
    ];
    this.bgRenderA = mkBG(this.pipeRender, 'RenderA', renderEntries(this.bufParA));
    this.bgRenderB = mkBG(this.pipeRender, 'RenderB', renderEntries(this.bufParB));

    // ── 粒子ソート ────────────────────────────────────────────────
    const sortInitEntries = (par) => [
      entry(0, buf(this.bufSortParams)),
      entry(1, buf(par)),
      entry(2, buf(this.bufCamera)),
      entry(3, buf(this.bufSortKey)),
      entry(4, buf(this.bufSortIdx)),
    ];
    this.bgSortInitA = mkBG(this.pipeSortInit, 'SortInitA', sortInitEntries(this.bufParA));
    this.bgSortInitB = mkBG(this.pipeSortInit, 'SortInitB', sortInitEntries(this.bufParB));
    this.bgSortStep  = mkBG(this.pipeSortStep, 'SortStep', [
      entry(0, buf(this.bufSortParams)),
      entry(1, buf(this.bufSortKey)),
      entry(2, buf(this.bufSortIdx)),
    ]);

    // ── 部屋ワイヤーフレーム ──────────────────────────────────────
    this.bgRoom = mkBG(this.pipeRoom, 'Room', [
      entry(0, buf(this.bufCamera)),
    ]);

    const sliceEntries = (par) => [
      entry(0, buf(this.bufParams)),
      entry(1, buf(par)),
      entry(2, buf(this.bufHashR)),
      entry(3, buf(this.bufHashD)),
      entry(4, buf(this.bufCamera)),
      entry(5, buf(this.bufSlice)),
    ];
    this.bgSliceA = mkBG(this.pipeSliceCompute, 'SliceComputeA', sliceEntries(this.bufParA));
    this.bgSliceB = mkBG(this.pipeSliceCompute, 'SliceComputeB', sliceEntries(this.bufParB));
    this.bgSliceRender = mkBG(this.pipeSliceRender, 'SliceRender', [
      entry(0, buf(this.bufCamera)),
      entry(1, buf(this.bufSlice)),
      entry(2, buf(this.bufParams)),
      entry(3, buf(this.bufCamera)),
    ]);
    this.bgNozzle = mkBG(this.pipeNozzle, 'NozzleRender', [entry(0, buf(this.bufCamera))]);
  }

  // ────────────────────────────────────────────────────────────────
  //  1ステップ計算
  // ────────────────────────────────────────────────────────────────
  _computeStep() {
    const dev = this.device;
    const N   = this.N;
    const WG  = 256;
    const nWG = Math.ceil(N / WG);
    const nWGH= Math.ceil(this.HASH_SIZE / WG);
    const pp  = this.pingPong;  // 0: A→B, 1: B→A

    const enc = dev.createCommandEncoder({ label: 'MPS Step' });

    // ── 1. ハッシュリセット ────────────────────────────────────────
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeHashZero);
      pass.setBindGroup(0, this.bgHashZeroA);
      pass.dispatchWorkgroups(nWGH);
      pass.end();
    }

    // ── 2. ハッシュ構築 ────────────────────────────────────────────
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeHashBuild);
      pass.setBindGroup(0, pp === 0 ? this.bgHashBuildA : this.bgHashBuildB);
      pass.dispatchWorkgroups(nWG);
      pass.end();
    }

    // ── 3. ハッシュカウントコピー (atomic → non-atomic) ────────────
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeHashCopy);
      pass.setBindGroup(0, this.bgHashCp);
      pass.dispatchWorkgroups(nWGH);
      pass.end();
    }

    // ── 4. 数密度・圧力計算 ────────────────────────────────────────
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeDensity);
      pass.setBindGroup(0, pp === 0 ? this.bgDensityA : this.bgDensityB);
      pass.dispatchWorkgroups(nWG);
      pass.end();
    }

    // pingPong切り替え (数密度はA→BまたはB→Aに書き込んだ)
    this.pingPong = 1 - pp;
    const pp2 = this.pingPong;

    // ── 5. 陽的速度・位置更新 (in=density出力, out=新バッファ) ─────
    // ここではhashは密度計算時と同じ入力粒子位置に基づいている
    // (pingPongを再度切り替えて同じinputを使う)
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeExplicit);
      pass.setBindGroup(0, pp2 === 0 ? this.bgExplicitA : this.bgExplicitB);
      pass.dispatchWorkgroups(nWG);
      pass.end();
    }
    this.pingPong = 1 - pp2;

    // ── 6. 温度輸送 ────────────────────────────────────────────────
    {
      const pp3 = this.pingPong;
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeTemp);
      pass.setBindGroup(0, pp3 === 0 ? this.bgTempA : this.bgTempB);
      pass.dispatchWorkgroups(nWG);
      pass.end();
      this.pingPong = 1 - pp3;
    }

    // ── 7. 流入境界条件の強制印加 ─────────────────────────────────
    {
      const pp4 = this.pingPong;
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeInletBC);
      pass.setBindGroup(0, pp4 === 0 ? this.bgInletA : this.bgInletB);
      pass.dispatchWorkgroups(nWG);
      pass.end();
    }

    dev.queue.submit([enc.finish()]);
    this.stepCount++;
    this.stats.step = this.stepCount;
    this.stats.time = this.stepCount * this.DT;
  }

  // ────────────────────────────────────────────────────────────────
  //  粒子ソート (カメラ距離、バイトニックソート)
  //  半透明ビルボードは描画順に結果が左右されるため、毎フレーム
  //  「カメラから遠い→近い」の順にインデックス配列を並べ替えてから描く。
  // ────────────────────────────────────────────────────────────────
  _sortParticles(enc, pp) {
    const dev = this.device;
    const sp  = this._sortParamsCPU;
    const nWG = Math.ceil(this.sortPaddedN / 256);

    // 1) 距離キー初期化 (今フレームの粒子位置・カメラ位置から)
    sp[0] = this.N; sp[1] = this.sortPaddedN; sp[2] = 0; sp[3] = 0;
    dev.queue.writeBuffer(this.bufSortParams, 0, sp);
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeSortInit);
      pass.setBindGroup(0, pp === 0 ? this.bgSortInitA : this.bgSortInitB);
      pass.dispatchWorkgroups(nWG);
      pass.end();
    }

    // 2) バイトニックソート本体 ((k,j)ステージごとに1ディスパッチ)
    for (const [k, j] of this._sortStages) {
      sp[2] = k; sp[3] = j;
      dev.queue.writeBuffer(this.bufSortParams, 0, sp);
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipeSortStep);
      pass.setBindGroup(0, this.bgSortStep);
      pass.dispatchWorkgroups(nWG);
      pass.end();
    }
  }

  // ────────────────────────────────────────────────────────────────
  //  レンダリング
  // ────────────────────────────────────────────────────────────────
  _render() {
    const dev = this.device;
    const cv  = this.canvas;

    // キャンバスサイズ確認・更新
    const w = Math.max(1, cv.clientWidth  | 0);
    const h = Math.max(1, cv.clientHeight | 0);
    if (cv.width !== w || cv.height !== h) {
      cv.width  = w;
      cv.height = h;
      this._createDepthTexture();
    }

    this._updateCameraBuffer();

    const tex  = this.context.getCurrentTexture();
    const view = tex.createView();
    const depthView = this.depthTex.createView();

    const enc = dev.createCommandEncoder({ label: 'Render' });
    // コンターモード(visualMode===2)のみ補間グリッドを計算。
    // 断面モード(visualMode===1)は粒子のみ描画するため不要。
    if (this.visualMode === 2) {
      const slicePass = enc.beginComputePass();
      slicePass.setPipeline(this.pipeSliceCompute);
      slicePass.setBindGroup(0, this.pingPong === 0 ? this.bgSliceA : this.bgSliceB);
      slicePass.dispatchWorkgroups(9, 9);  // ceil(65/8) = 9 (workgroup_size 8x8)
      slicePass.end();
    }
    // コンターモード(visualMode===2)は粒子を描かないのでソートも省略
    if (this.visualMode !== 2) {
      this._sortParticles(enc, this.pingPong);
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue : { r: 0.05, g: 0.05, b: 0.12, a: 1 },
        loadOp     : 'clear',
        storeOp    : 'store',
      }],
      depthStencilAttachment: {
        view       : depthView,
        depthClearValue  : 1,
        depthLoadOp      : 'clear',
        depthStoreOp     : 'store',
      },
    });

    // ── 部屋ワイヤーフレーム ──────────────────────────────────────
    pass.setPipeline(this.pipeRoom);
    pass.setBindGroup(0, this.bgRoom);
    pass.setVertexBuffer(0, this.bufRoom);
    pass.setIndexBuffer(this.bufRoomIdx, 'uint16');
    pass.drawIndexed(24);

    pass.setPipeline(this.pipeNozzle);
    pass.setBindGroup(0, this.bgNozzle);
    pass.setVertexBuffer(0, this.bufNozzle);
    pass.draw(this.nozzleVertexCount);

    // 断面サーフェス描画: コンターモードのみ。
    // 断面モード(visualMode===1)は粒子ドットのみ描画するためスキップ。
    if (this.visualMode === 2) {
      pass.setPipeline(this.pipeSliceRender);
      pass.setBindGroup(0, this.bgSliceRender);
      pass.draw(64 * 64 * 6);  // 64x64セル (65x65グリッド)
    }

    // ── 粒子描画 ──────────────────────────────────────────────────
    // コンターモード(visualMode===2)では補間グリッド(pipeSliceRender)のみで
    // 温度分布を滑らかに表現する。個々の粒子ビルボードを重ねて描画すると
    // 画像のように点状ノイズが乗ってしまうため、このモードでは描画しない。
    if (this.visualMode !== 2) {
      const pp = this.pingPong;  // 最後に書き込まれたバッファ
      pass.setPipeline(this.pipeRender);
      pass.setBindGroup(0, pp === 0 ? this.bgRenderA : this.bgRenderB);
      // 4頂点/粒子 (triangle-strip quad), N インスタンス
      pass.draw(4, this.N, 0, 0);
    }

    pass.end();
    dev.queue.submit([enc.finish()]);
  }

  // ────────────────────────────────────────────────────────────────
  //  アニメーションフレーム (substeps + render)
  // ────────────────────────────────────────────────────────────────
  tick() {
    if (!this.paused) {
      this._updateParamsBuffer();
      for (let s = 0; s < this.substepsPerFrame && this.stepCount < this.steadySteps; s++) {
        this._computeStep();
      }
      if (this.steadyMode && this.stepCount >= this.steadySteps) {
        this.paused = true;
        this.steadyComplete = true;
      }
    }
    this._render();
  }

  updateView() {
    const signature = `${this.visualMode}:${this.sliceMode}`;
    if (signature !== this.viewSignature) {
      if (this.visualMode !== 0) {
        this.camera.theta = this.sliceMode === 1 ? Math.PI / 2 : 0;
        this.camera.phi = Math.PI / 2 - 0.04;
      } else {
        this.camera.theta = 0.4;
        this.camera.phi = 0.5;
      }
      this.viewSignature = signature;
    }
    this._updateCameraBuffer();
    this._updateParamsBuffer();
  }

  // ────────────────────────────────────────────────────────────────
  //  ノズル切り替え (再初期化)
  // ────────────────────────────────────────────────────────────────
  async setNozzle(type) {
    this.paused = true;
    this.steadyComplete = false;
    this.nozzleType = type;
    this.stepCount = 0;
    this.pingPong = 0;
    await this._createBuffers(type);
    this._createBindGroups();
    this._updateParamsBuffer();
    this.paused = false;
  }

  // ────────────────────────────────────────────────────────────────
  //  吹出し条件の変更 (風速 / 温度)
  // ────────────────────────────────────────────────────────────────
  setInletCondition(velocity, temp) {
    if (velocity !== undefined && !isNaN(velocity)) AC.inletVelocity = velocity;
    if (temp !== undefined && !isNaN(temp)) {
      AC.inletTemp   = temp;
      this.T_IN      = temp;
      this.stats.tMin = temp;
    }
    updateNozzleVelocities();
    if (this.device && this.bufParams) this._updateParamsBuffer();
  }

  // ────────────────────────────────────────────────────────────────
  //  リセット (粒子配置を初期状態に戻す)
  // ────────────────────────────────────────────────────────────────
  async reset() {
    this.paused = true;
    this.stepCount = 0;
    this.pingPong = 0;
    this.steadyComplete = false;
    await this.device.queue.onSubmittedWorkDone();
    await this._createBuffers(this.nozzleType);
    this._createBindGroups();
    this._updateParamsBuffer();
  }

  // ────────────────────────────────────────────────────────────────
  //  破棄
  // ────────────────────────────────────────────────────────────────
  destroy() {
    [
      this.bufParA, this.bufParB, this.bufHash, this.bufHashR, this.bufHashD,
      this.bufParams, this.bufCamera, this.bufRoom, this.bufRoomIdx,
      this.bufSlice, this.bufNozzle,
    ].forEach(b => b?.destroy());
    this.depthTex?.destroy();
    this.device?.destroy?.();
  }
}

// ══════════════════════════════════════════════════════════════════
//  WebGPUFVM — 統合有限体積法(FVM)ソルバー (本番用)
//
//  構成は cpu_fallback.js と同じ:
//    室内   z ∈ [0, DOM]           … NX³ の非均一 Cartesian 格子 (厳密な立方体)
//    ノズル z ∈ [DOM, DOM+0.18]    … 天井の上に積む微細セル (step = 0.18/48)
//  天井(k=NX-1)の開口セルとノズル最下層を _createUnifiedNeighbors() で
//  接続し、1つの非構造セル隣接リストとして解く。
//
//  ※ 本クラスのコンストラクタはファイル破損により失われていたため復元した。
//    物理定数は cpu_fallback.js (CPUモード) と一致させてある。両モードの
//    結果を突き合わせる前提なので、変更する場合は両方を同時に直すこと。
// ══════════════════════════════════════════════════════════════════
class WebGPUFVM {
  constructor() {
    // ── 領域・格子 ────────────────────────────────────────────────
    this.DOM      = 2.0;              // 室内一辺 [m] (2×2×2 の立方体)
    this.NOZZLE_H = 0.18;             // ノズル高さ [m] (室内の「上」に積む)
    this.DOM_Z    = this.DOM + this.NOZZLE_H;  // z方向総高さ [m] = 2.18
    this.NX       = 64;               // 室内格子解像度 (NX³ = 262,144セル)
    this.N        = this.NX ** 3;     // _createBuffers() でノズルセル分が加算される
    this.H        = this.DOM / this.NX;  // 均一格子換算の参照幅 [m]

    // ── 物理定数 (cpu_fallback.js と同値) ──────────────────────────
    this.DT     = 0.0001;   // 時間刻み [s]
    this.NU     = 0.005;    // 乱流有効粘度 [m²/s]
    this.ALPHA  = 0.005;    // 乱流有効熱拡散率 [m²/s]
    this.G      = 9.81;     // 重力 [m/s²]
    this.BETA   = 1 / 307;  // 体膨張係数 (Boussinesq近似)
    this.T_REF  = 34.0;     // 室内初期温度・壁温 [°C]
    this.T_IN   = 23.5;     // 吹出し温度 [°C]
    this.RHO0   = 1.2;      // 空気密度 [kg/m³]
    this.PRESSURE_ITERS = 32;  // 圧力ポアソン方程式のヤコビ反復回数/ステップ

    // ── セルタイプ (state.w に格納) ────────────────────────────────
    this.FLUID = 0; this.WALL = 1; this.INLET = 2;

    // ── エンジン識別 (sim_mode.js の手動切替で参照) ─────────────────
    this.engineMode  = 'gpu';
    this.engineLabel = 'GPU (WebGPU / FVM)';

    // ── WebGPU リソース ───────────────────────────────────────────
    this.device = null; this.canvas = null; this.context = null;
    this.format = 'bgra8unorm';
    this.nozzleType = 1;

    // 格子面座標 (_initGrid で生成)
    this.xg = null; this.yg = null; this.zg = null;

    // ノズル微細セル
    this.nozzleCount = 0; this.nozzleStateData = null; this.nozzleBottomIndices = [];
    this.gridVertexCount = 0; this.nozzleWireVertexCount = 0; this.roomGridVertexCount = 0;

    // バッファ
    this.bufStateA = this.bufStateB = this.bufUStar = this.bufDiv = null;
    this.bufPressureA = this.bufPressureB = null;
    this.bufParams = this.bufCamera = this.bufSortIdx = null;
    this.bufNozzleState = this.bufNozzleIdx = this.bufNozzleConnection = null;
    this.bufNeighbors = this.bufCellH = this.bufGridAxes = null;
    this.bufGrid = this.bufNozzleWire = this.bufSlice = null;

    // パイプライン
    this.pipePredict = this.pipeDivergence = this.pipePressure = this.pipeCorrect = this.pipeTemperature = null;
    this.pipeRender=this.pipeRoom=this.pipeNozzle=this.pipeSliceCompute=this.pipeSliceRender=this.pipeVector=null; this.pipeSliceCell=null; this.bgSliceCellA=this.bgSliceCellB=null; this.bgPredict=this.bgDivergence=null; this.bgGrid=this.bgSliceCompute=this.bgSliceRender=this.bgVector=null;
    this.bgPressureAB=this.bgPressureBA=this.bgCorrect=this.bgTemperature=null; this.bgRenderA=this.bgRenderB=null;
    this.bgRoom=this.bgNozzle=null; this.stepCount=0; this.stateIndex=0; this.paused=true;
    this.displayMode=1; this.visualMode=1; this.sliceMode=1; this.slicePosition=1; this.viewSignature='';
    this.nozzleFocus=true; // ノズル部拡大トグル(断面・コンター表示時のみ有効)
    this.substepsPerFrame=8; this.steadyMode=false; this.steadyComplete=false;
    this.stats={tMin:this.T_IN,tMax:this.T_REF,vMax:0,step:0,time:0};
    this.camera={theta:Math.PI/2,phi:Math.PI/2-0.04,radius:0.7,target:[0,(this.DOM+this.DOM_Z)/2-this.DOM/2,0]};
  }
  async init(canvas,nozzleType=1){this.canvas=canvas;this.nozzleType=nozzleType;if(!navigator.gpu)throw Error('WebGPU未対応ブラウザです');const a=await navigator.gpu.requestAdapter();if(!a)throw Error('WebGPUアダプター取得失敗');this.device=await a.requestDevice();this.format=navigator.gpu.getPreferredCanvasFormat();canvas.width=canvas.width||canvas.clientWidth||800;canvas.height=canvas.height||canvas.clientHeight||600;this.context=canvas.getContext('webgpu');this.context.configure({device:this.device,format:this.format,alphaMode:'premultiplied'});await this._createBuffers();await this._createPipelines();this._createBindGroups();this._setupCamera();this.updateView();return this;}
  _idx(i,j,k){return i+this.NX*(j+this.NX*k);}
  _initGrid(){
    // BETA_XY: 1.8→2.2, BETA_ZT: 2.5→3.5, ALPHA_Z: (旧固定0.7)→0.85 に変更。
    //
    // 理由: ノズル微細格子の刻み幅は step=0.18/48≈3.75mm。従来設定では
    //   室内側の天井直下セル厚が≈15.0mm、中心XYセルが≈6.5mmと、ノズル側
    //   (3.75mm)に対し最大4倍の解像度差があった。_createNozzleConnection()
    //   はこの粗い室内セルと細かいノズルセルを1対1で接続するため、境界での
    //   質量・運動量の受け渡しが局所的に偏り、ノズル部の数値的発散の一因に
    //   なっていた。
    //   z方向は betaT を上げるだけでは天井直下セル厚が≈13.4mmで頭打ちになる
    //   (buildStretchedZGrid内部の天井/床ブレンド比が固定だったため)ため、
    //   ブレンド比 alpha を引数化し0.85に引き上げることで≈7.1mmまで縮小。
    //   これにより天井直下セル≈7mm・中心XYセル≈3.6mmとなり、ノズル微細セル
    //   (3.75mm)とほぼ同スケールに揃う。
    //   トレードオフとして床側セルは≈60mm→≈95mmとやや粗くなるが、今回の
    //   発散箇所(ノズル近傍)とは無関係な領域なので許容している。
    const BETA_XY=2.2, BETA_ZT=3.5, BETA_ZF=1.2, ALPHA_Z=0.85, cx=this.DOM/2, cy=this.DOM/2;
    this.xg=buildStretchedGrid(this.NX+1,this.DOM,cx,BETA_XY);
    this.yg=buildStretchedGrid(this.NX+1,this.DOM,cy,BETA_XY);
    this.zg=buildStretchedZGrid(this.NX+1,this.DOM,BETA_ZT,BETA_ZF,ALPHA_Z);
  }
  _sampleNozzle(nozzle,core,x,y,z,localH){const sub=12,h=localH||this.H,half=h/2,zBot=this.DOM-.18,cx=this.DOM/2;let ni=0,nw=0,no=0,total=0;for(let iz=0;iz<3;iz++){const zp=Math.min(.18,Math.max(0,z+(iz-1)*half*.9-zBot)),sec=getNozzleSection(this.nozzleType,zp),ic=core&&zp>=core.zStart&&zp<=core.zEnd;for(let sx=0;sx<sub;sx++)for(let sy=0;sy<sub;sy++){const px=x-half+(sx+.5)/sub*h,py=y-half+(sy+.5)/sub*h,r=Math.hypot(px-cx,py-cx);total++;if(r>sec.rOuter)no++;else if(r>=sec.rInner||(ic&&r<core.rOut))nw++;else ni++;}}return{inletFrac:ni/total,wallFrac:nw/total,outsideFrac:no/total};}
  _initialData(){
    this._initGrid();
    const raw=new Float32Array(this.N*16),n=NOZZLES[this.nozzleType];
    const outletR=n.outlet.innerR;
    for(let i=0;i<this.NX;i++)for(let j=0;j<this.NX;j++)for(let k=0;k<this.NX;k++){
      const c=this._idx(i,j,k),x=(this.xg[i]+this.xg[i+1])*0.5,y=(this.yg[j]+this.yg[j+1])*0.5,z=(this.zg[k]+this.zg[k+1])*0.5;
      const side=i===0||i===this.NX-1||j===0||j===this.NX-1;
      const topWall=k===this.NX-1;
      const r=Math.hypot(x-1,y-1);
      const inOutlet=topWall && (r<=outletR);
      let type=this.FLUID,vz=(Math.random()-.5)*.02,t=this.T_REF;
      if(side||k===0||(topWall&&!inOutlet)){
        type=this.WALL;
      }else if(inOutlet){
        type=this.INLET;
        vz=-n.outlet.velocity;
        t=this.T_IN;
      }
      const b=c*16;raw[b]=x;raw[b+1]=y;raw[b+2]=z;raw[b+3]=type;raw[b+6]=vz;raw[b+8]=t;raw[b+10]=Math.abs(vz);
    }
    return raw;
  }
  _storage(label,size){return this.device.createBuffer({label,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});}
  _createRoomNeighbors(){
    const neighbors=new Int32Array(this.N*6);
    const index=(i,j,k)=>i+this.NX*(j+this.NX*k);
    for(let k=0;k<this.NX;k++)for(let j=0;j<this.NX;j++)for(let i=0;i<this.NX;i++){
      const c=index(i,j,k),base=c*6;
      neighbors[base]=i+1<this.NX?index(i+1,j,k):c;
      neighbors[base+1]=i>0?index(i-1,j,k):c;
      neighbors[base+2]=j+1<this.NX?index(i,j+1,k):c;
      neighbors[base+3]=j>0?index(i,j-1,k):c;
      neighbors[base+4]=k+1<this.NX?index(i,j,k+1):c;
      neighbors[base+5]=k>0?index(i,j,k-1):c;
    }
    this.bufNeighbors?.destroy();
    this.bufNeighbors=this._storage('FVM room cell neighbors',neighbors.byteLength);
    this.device.queue.writeBuffer(this.bufNeighbors,0,neighbors);
  }
  _createCellDistBuffer(){
    const NX=this.NX, roomN=NX**3, total=this.N, dist=new Float32Array(total*8);
    const xg=this.xg, yg=this.yg, zg=this.zg;
    const xc=new Float32Array(NX), yc=new Float32Array(NX), zc=new Float32Array(NX);
    for(let i=0;i<NX;i++) xc[i]=(xg[i]+xg[i+1])*0.5;
    for(let j=0;j<NX;j++) yc[j]=(yg[j]+yg[j+1])*0.5;
    for(let k=0;k<NX;k++) zc[k]=(zg[k]+zg[k+1])*0.5;

    for(let k=0;k<NX;k++)for(let j=0;j<NX;j++)for(let i=0;i<NX;i++){
      const c=this._idx(i,j,k);
      const hxE = i < NX - 1 ? (xc[i+1] - xc[i]) : (xg[NX] - xc[NX-1]);
      const hxW = i > 0 ? (xc[i] - xc[i-1]) : (xc[0] - xg[0]);
      const hyN = j < NX - 1 ? (yc[j+1] - yc[j]) : (yg[NX] - yc[NX-1]);
      const hyS = j > 0 ? (yc[j] - yc[j-1]) : (yc[0] - yg[0]);
      const hzU = k < NX - 1 ? (zc[k+1] - zc[k]) : (zg[NX] - zc[NX-1]);
      const hzD = k > 0 ? (zc[k] - zc[k-1]) : (zc[0] - zg[0]);
      const base=c*8;
      dist[base]=hxE; dist[base+1]=hxW; dist[base+2]=hyN; dist[base+3]=hyS;
      dist[base+4]=hzU; dist[base+5]=hzD; dist[base+6]=0; dist[base+7]=0;
    }
    const step=.18/48;
    for(let n=0;n<this.nozzleCount;n++){
      const base=(roomN+n)*8;
      dist[base]=step; dist[base+1]=step; dist[base+2]=step; dist[base+3]=step;
      dist[base+4]=step; dist[base+5]=step; dist[base+6]=0; dist[base+7]=0;
    }
    this.bufCellH?.destroy();
    this.bufCellH=this._storage('FVM cell local distances',dist.byteLength);
    this.device.queue.writeBuffer(this.bufCellH,0,dist);
  }
  async _createBuffers(){
    const d=this.device;this.N=this.NX**3;const data=this._initialData();
    this.bufStateA?.destroy();this.bufStateB?.destroy();
    this.bufStateA=this._storage('FVM State A',data.byteLength);this.bufStateB=this._storage('FVM State B',data.byteLength);
    d.queue.writeBuffer(this.bufStateA,0,data);d.queue.writeBuffer(this.bufStateB,0,data);
    this._createRoomNeighbors();
    this.bufSlice=this._storage('FVM Slice',65*65*4);
    this.bufParams=d.createBuffer({size:128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.bufCamera=d.createBuffer({size:96,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this._updateParamsBuffer();this._updateCameraBuffer();
    // 格子ワイヤーフレームはノズルセルの配置を参照するため、
    // _createNozzleState() の後に生成する。
    this._createNozzleWire();this._createNozzleState();this._createGridMesh();
    this._mergeNozzleCells(data);
    this._createCellDistBuffer();
    this.bufUStar=this._storage('FVM UStar',this.N*16);
    this.bufDiv=this._storage('FVM Divergence',this.N*4);
    this.bufPressureA=this._storage('FVM Pressure A',this.N*4);
    this.bufPressureB=this._storage('FVM Pressure B',this.N*4);
    this._createNozzleConnection(data);
    const axes=new Float32Array((this.NX+1)*3);
    axes.set(this.xg,0);
    axes.set(this.yg,this.NX+1);
    axes.set(this.zg,(this.NX+1)*2);
    this.bufGridAxes?.destroy();
    this.bufGridAxes=this._storage('FVM grid axes coords',axes.byteLength);
    this.device.queue.writeBuffer(this.bufGridAxes,0,axes);
    this._updateParamsBuffer();
    this.bufSortIdx=this._storage('Unified FVM indices',this.N*4);
    const ids=new Uint32Array(this.N);for(let i=0;i<this.N;i++)ids[i]=i;
    d.queue.writeBuffer(this.bufSortIdx,0,ids);
  }
  _createGridMesh(){
    // ── 室内 + ノズルを通して隙間のない1つの格子ラティスを作る ──────────
    // CPUモード(cpu_fallback.js の _updateVisuals)と同じ「セル辺の共有方式」に
    // 合わせる。すなわち各セルは隣接セルと辺を共有し、格子全体が途切れなく
    // 繋がる。ワールド座標の対応も CPU と同一:
    //     X = x - DOM/2,  Z = y - DOM/2,  Y(鉛直) = z - DOM/2
    //     室内   z ∈ [0, 2.0]    → Y ∈ [-1.0, +1.0]
    //     ノズル z ∈ [2.0, 2.18] → Y ∈ [+1.0, +1.18]
    //
    // 室内は全セルが存在するので、セル辺を1本ずつ出す代わりに同じ線集合を
    // 「共線分をまとめた全長線」として出す(見た目は完全に同一で、頂点数が
    //  3・NX³・2 ≈ 157万 → 3・(NX+1)²・2 ≈ 2.5万 に減る)。
    // ノズル区間はセルが円形に欠けるため、セル辺を張ってから辺キーで重複を
    // 除去する。室内の天井面(Y=+1)とノズル最下面はぴったり一致するので
    // 継ぎ目に隙間は出ない。
    const lines=[], n=this.NX, add=(a,b)=>lines.push(...a,...b);
    const xg=this.xg, yg=this.yg, zg=this.zg, o=this.DOM/2;

    // 室内: 鉛直(Y) / X方向 / Z方向 の全長線
    for(let i=0;i<=n;i++)for(let j=0;j<=n;j++) add([xg[i]-o,-o,yg[j]-o],[xg[i]-o,o,yg[j]-o]);
    for(let k=0;k<=n;k++)for(let j=0;j<=n;j++) add([-o,zg[k]-o,yg[j]-o],[o,zg[k]-o,yg[j]-o]);
    for(let i=0;i<=n;i++)for(let k=0;k<=n;k++) add([xg[i]-o,zg[k]-o,-o],[xg[i]-o,zg[k]-o,o]);

    // 室内分の頂点数を記録しておく (断面・コンター表示時にノズル部分の
    // メッシュだけを描き分けるための境界。ノズル部はベクトルを出さず
    // メッシュのみで示す方針のため、この頂点区間だけを別途描画する)
    this.roomGridVertexCount = lines.length / 3;

    // ノズル区間: 実在するセルだけセル辺を張る (重複辺は除去)
    const d=this.nozzleStateData;
    if(d && this.nozzleCount){
      const step=this.NOZZLE_H/48, h=step*0.5, seen=new Set();
      const q=v=>Math.round(v/h);   // 端点は必ず h の整数倍に乗る
      const edge=(a,b)=>{
        const ka=`${q(a[0])},${q(a[1])},${q(a[2])}`, kb=`${q(b[0])},${q(b[1])},${q(b[2])}`;
        const key = ka<kb ? ka+'|'+kb : kb+'|'+ka;
        if(seen.has(key)) return;
        seen.add(key);
        add(a,b);
      };
      const E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      for(let c=0;c<this.nozzleCount;c++){
        const b=c*16;
        const X=d[b]-o, Z=d[b+1]-o, Y=d[b+2]-o;
        const x0=X-h,x1=X+h, z0=Z-h,z1=Z+h, y0=Y-h,y1=Y+h;
        const v=[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                 [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]];
        for(const [p0,p1] of E) edge(v[p0],v[p1]);
      }
    }

    const data=new Float32Array(lines);
    this.bufGrid?.destroy();
    this.bufGrid=this.device.createBuffer({label:'FVM wireframe grid',size:data.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
    this.device.queue.writeBuffer(this.bufGrid,0,data);
    this.gridVertexCount=data.length/3;
  }
  _createNozzleWire(){
    // getNozzleWireframeLines で外管および内部二重管・整流筒の全輪郭線を統一生成
    // Three.js座標系では y=1.0 が天井(ノズル底端)
    const lines = getNozzleWireframeLines(this.nozzleType, 1.0);
    const data = new Float32Array(lines);
    this.bufNozzleWire?.destroy();
    this.bufNozzleWire = this.device.createBuffer({
      label: 'Nozzle fine wireframe',
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(this.bufNozzleWire, 0, data);
    this.nozzleWireVertexCount = data.length / 3;
  }
  _createNozzleState(){
    const nozzle = NOZZLES[this.nozzleType], step = .18 / 48, raw = [];
    this.nozzleBottomIndices = [];
    for(let z = this.DOM + step / 2; z < this.DOM + .18; z += step){
      const zp = z - this.DOM, sec = getNozzleSection(this.nozzleType, zp);
      const inner = getNozzleInnerStructure(this.nozzleType, zp);
      let blockedArea = 0;
      if (inner.hasInner) {
        blockedArea = Math.PI * (inner.rOut * inner.rOut - inner.rIn * inner.rIn);
      }
      const flowArea = Math.max(Math.PI * (sec.rInner * sec.rInner) - blockedArea, 1e-8);
      const localV = Math.min(AC.inletVelocity * AC.inletArea / flowArea, 35);

      for(let x = 1 - sec.rOuter; x <= 1 + sec.rOuter; x += step) {
        for(let y = 1 - sec.rOuter; y <= 1 + sec.rOuter; y += step){
          const r = Math.hypot(x - 1, y - 1);
          if (r > sec.rOuter) continue;
          const isWall = isNozzleWall(this.nozzleType, zp, r);
          let type = isWall ? 1 : 0;
          if (type === 0 && zp >= .18 - step * 1.2) {
            type = 2; // 最上面流入面はINLETとして固定供給 (中空内部および外側環状流路)
          }
          const index = raw.length / 16;
          if (zp < step && type === 0) this.nozzleBottomIndices.push(index);
          raw.push(x, y, z, type, 0, 0, type === 1 ? 0 : -localV, 0, type === 1 ? this.T_REF : this.T_IN, 0, type === 1 ? 0 : localV, 0, 0, 0, 0, 0);
        }
      }
    }
    this.nozzleCount = raw.length / 16;
    this.nozzleStateData = new Float32Array(raw);
    this.bufNozzleState?.destroy();
    this.bufNozzleIdx?.destroy();
    this.bufNozzleState = this._storage('Nozzle fine distribution', this.nozzleStateData.byteLength);
    this.bufNozzleIdx = this._storage('Nozzle fine indices', this.nozzleCount * 4);
    this.device.queue.writeBuffer(this.bufNozzleState, 0, this.nozzleStateData);
    const ids = new Uint32Array(this.nozzleCount);
    for(let i = 0; i < this.nozzleCount; i++) ids[i] = i;
    this.device.queue.writeBuffer(this.bufNozzleIdx, 0, ids);
  }
  _mergeNozzleCells(roomData){const roomN=this.N,total=roomN+this.nozzleCount,combined=new Float32Array(total*16);combined.set(roomData);combined.set(this.nozzleStateData,roomN*16);this.N=total;this.bufStateA?.destroy();this.bufStateB?.destroy();this.bufStateA=this._storage('Unified FVM State A',combined.byteLength);this.bufStateB=this._storage('Unified FVM State B',combined.byteLength);this.device.queue.writeBuffer(this.bufStateA,0,combined);this.device.queue.writeBuffer(this.bufStateB,0,combined);this._createUnifiedNeighbors();}
  _createUnifiedNeighbors(){
    const neighbors=new Int32Array(this.N*6);const roomN=this.NX**3,index=(i,j,k)=>i+this.NX*(j+this.NX*k);
    for(let c=0;c<roomN;c++){
      const k=Math.floor(c/(this.NX*this.NX)),r=c%(this.NX*this.NX),j=Math.floor(r/this.NX),i=r%this.NX,b=c*6;
      neighbors[b]=i+1<this.NX?index(i+1,j,k):c;
      neighbors[b+1]=i>0?index(i-1,j,k):c;
      neighbors[b+2]=j+1<this.NX?index(i,j+1,k):c;
      neighbors[b+3]=j>0?index(i,j-1,k):c;
      neighbors[b+4]=k+1<this.NX?index(i,j,k+1):c;
      neighbors[b+5]=k>0?index(i,j,k-1):c;
    }
    const nozzleBase=roomN,step=.18/48,lookup=new Map();
    for(let n=0;n<this.nozzleCount;n++){const b=n*16,key=`${Math.round(this.nozzleStateData[b]/step)},${Math.round(this.nozzleStateData[b+1]/step)},${Math.round((this.nozzleStateData[b+2]-this.DOM)/step)}`;lookup.set(key,n+nozzleBase);}
    const near=(x,y,z)=>lookup.get(`${Math.round(x/step)},${Math.round(y/step)},${Math.round((z-this.DOM)/step)}`);
    for(let n=0;n<this.nozzleCount;n++){
      const b=n*16,c=nozzleBase+n,x=this.nozzleStateData[b],y=this.nozzleStateData[b+1],z=this.nozzleStateData[b+2],base=c*6;
      neighbors[base]=near(x+step,y,z)??c;
      neighbors[base+1]=near(x-step,y,z)??c;
      neighbors[base+2]=near(x,y+step,z)??c;
      neighbors[base+3]=near(x,y-step,z)??c;
      neighbors[base+4]=near(x,y,z+step)??c;
      neighbors[base+5]=near(x,y,z-step)??c;
    }
    // 先にノズル内部の隣接関係を修復
    this._repairNozzleNeighbors(neighbors,roomN,step);

    // ★重要: _repairNozzleNeighbors の後にノズル最下層(出口)と室内天井セルの双方向接続を確立する。
    // 以前は _repairNozzleNeighbors が後に実行されていたため、最下層(iz=0)の下隣(スロット5)が
    // safe(nearest(-1, ...), c) = c (自分自身) に上書きされ、ノズル出口が完全に閉塞して
    // 上から吹き込む流体の逃げ場がなくなりノズル内で激しい数値発散・振動を起こしていた。
    for(const c of this.nozzleBottomIndices){
      const b=(nozzleBase+c)*6;
      const nx=this.nozzleStateData[c*16], ny=this.nozzleStateData[c*16+1];
      const ix=Math.max(1,Math.min(this.NX-2,bisectLeft(this.xg,nx)));
      const iy=Math.max(1,Math.min(this.NX-2,bisectLeft(this.yg,ny)));
      const roomCell=index(ix,iy,this.NX-1);
      neighbors[b+5]=roomCell;
      const rb=roomCell*6;
      neighbors[rb+4]=nozzleBase+c;
    }
    this.bufNeighbors?.destroy();
    this.bufNeighbors=this._storage('Unified unstructured cell neighbors',neighbors.byteLength);
    this.device.queue.writeBuffer(this.bufNeighbors,0,neighbors);
  }
  _repairNozzleNeighbors(neighbors,roomN,step){
    const layers=Array.from({length:49},()=>[]);
    for(let n=0;n<this.nozzleCount;n++){
      const b=n*16;
      const iz=Math.max(0,Math.min(48,Math.round((this.nozzleStateData[b+2]-this.DOM)/step)));
      layers[iz].push(n);
    }
    const nearest=(iz,x,y,isFluid)=>{
      if(iz<0||iz>=layers.length||!layers[iz].length)return -1;
      let best=-1,bestD=Infinity;
      for(const n of layers[iz]){
        const b=n*16;
        if(isFluid && this.nozzleStateData[b+3]===1) continue;
        const dx=this.nozzleStateData[b]-x,dy=this.nozzleStateData[b+1]-y,d=dx*dx+dy*dy;
        if(d<bestD){bestD=d;best=n;}
      }
      if(best>=0 && bestD<=step*step*2.5) return roomN+best;
      for(const n of layers[iz]){
        const b=n*16,dx=this.nozzleStateData[b]-x,dy=this.nozzleStateData[b+1]-y,d=dx*dx+dy*dy;
        if(d<bestD){bestD=d;best=n;}
      }
      return bestD<=step*step*2.5?roomN+best:-1;
    };
    const safe=(v,c)=>v>=0?v:c;
    for(let n=0;n<this.nozzleCount;n++){
      const b=n*16,c=roomN+n,x=this.nozzleStateData[b],y=this.nozzleStateData[b+1],z=this.nozzleStateData[b+2];
      const isFluid = this.nozzleStateData[b+3] !== 1;
      const iz=Math.max(0,Math.min(48,Math.round((z-this.DOM)/step))),base=c*6;
      neighbors[base]=safe(nearest(iz,x+step,y,isFluid),c);
      neighbors[base+1]=safe(nearest(iz,x-step,y,isFluid),c);
      neighbors[base+2]=safe(nearest(iz,x,y+step,isFluid),c);
      neighbors[base+3]=safe(nearest(iz,x,y-step,isFluid),c);
      neighbors[base+4]=safe(nearest(iz+1,x,y,isFluid),c);
      if(iz > 0){
        neighbors[base+5]=safe(nearest(iz-1,x,y,isFluid),c);
      }
    }
  }
  _createNozzleConnection(roomData){
    const roomN=this.NX**3,base=roomN,links=new Float32Array(this.N*4),bottom=this.nozzleBottomIndices||[];
    const kTop=this.NX-1;
    for(let i=0;i<this.NX;i++)for(let j=0;j<this.NX;j++){
      const c=this._idx(i,j,kTop),b=c*16;
      if(roomData[b+3]!==this.INLET||!bottom.length)continue;
      const x=roomData[b],y=roomData[b+1];
      let nearest=bottom[0],best=Infinity;
      for(const index of bottom){
        const n=index*16,dx=this.nozzleStateData[n]-x,dy=this.nozzleStateData[n+1]-y,d=dx*dx+dy*dy;
        if(d<best){best=d;nearest=index;}
      }
      const link=c*4;
      links[link]=base+nearest;
      links[link+1]=1.0;
      links[link+2]=1.0;
      links[link+3]=1.0;
    }
    this.bufNozzleConnection?.destroy();
    this.bufNozzleConnection=this._storage('Room-to-nozzle connection',links.byteLength);
    this.device.queue.writeBuffer(this.bufNozzleConnection,0,links);
  }
  _updateParamsBuffer(){const n=NOZZLES[this.nozzleType],b=new ArrayBuffer(128),u=new Uint32Array(b),f=new Float32Array(b);u[0]=this.N;u[1]=this.NX;u[2]=this.stepCount;f[4]=this.H;f[9]=this.DT;f[10]=this.NU;f[11]=this.ALPHA;f[12]=this.G;f[13]=this.BETA;f[14]=this.T_REF;f[15]=this.T_IN;f[16]=this.RHO0;f[19]=this.DOM;f[20]=n.outlet.velocity;u[21]=this.nozzleType;f[22]=AC.inletRadius;f[23]=1;f[24]=1;u[25]=n.outlet.hasCore?1:0;f[26]=n.outlet.coreR;f[27]=this.T_IN;f[28]=this.T_REF;f[29]=Math.max(n.outlet.velocity,AC.inletVelocity)*1.5;u[30]=this.displayMode;f[31]=this.DOM+.18;this.device.queue.writeBuffer(this.bufParams,0,b);}
  _updateCameraBuffer(){
    const {theta,phi,radius,target}=this.camera;
    const eye=[target[0]+radius*Math.sin(phi)*Math.sin(theta),target[1]+radius*Math.cos(phi),target[2]+radius*Math.sin(phi)*Math.cos(theta)];
    const f=[target[0]-eye[0],target[1]-eye[1],target[2]-eye[2]],fl=Math.hypot(...f);f[0]/=fl;f[1]/=fl;f[2]/=fl;
    const r=[f[2],0,-f[0]],rl=Math.hypot(...r);r[0]/=rl;r[2]/=rl;
    const u=[r[2]*f[1],r[0]*f[2]-r[2]*f[0],-r[0]*f[1]];
    const view=[r[0],u[0],-f[0],0,r[1],u[1],-f[1],0,r[2],u[2],-f[2],0,-r[0]*eye[0]-r[1]*eye[1]-r[2]*eye[2],-u[0]*eye[0]-u[1]*eye[1]-u[2]*eye[2],f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2],1];
    const aspect=Math.max(1,this.canvas.width/this.canvas.height),q=1/Math.tan(Math.PI/6),near=.1,far=100;
    const proj=[q/aspect,0,0,0,0,q,0,0,0,0,far/(near-far),-1,0,0,near*far/(near-far),0],m=new Float32Array(16);
    for(let col=0;col<4;col++)for(let row=0;row<4;row++){let sum=0;for(let k=0;k<4;k++)sum+=proj[k*4+row]*view[col*4+k];m[col*4+row]=sum;}
    const yz = (this.sliceMode === 1);
    const grid = yz ? this.xg : this.yg;
    const fIdx = (grid && grid.length) ? Math.max(1, Math.min(this.NX - 2, bisectLeft(grid, this.slicePosition))) : Math.floor(this.NX / 2);
    const buffer=new ArrayBuffer(96),values=new Float32Array(buffer);values.set(m);values[16]=eye[0];values[17]=eye[1];values[18]=eye[2];values[19]=fIdx;values[20]=this.visualMode?this.sliceMode:0;values[21]=this.slicePosition;values[22]=this.H*.8;values[23]=this.visualMode;this.device.queue.writeBuffer(this.bufCamera,0,buffer);
  }
  _setupCamera(){
    let drag=null;const c=this.canvas;
    c.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,theta:this.camera.theta,phi:this.camera.phi};c.setPointerCapture(e.pointerId);});
    c.addEventListener('pointermove',e=>{if(!drag)return;this.camera.theta=drag.theta-(e.clientX-drag.x)*.005;this.camera.phi=Math.max(.05,Math.min(Math.PI-.05,drag.phi+(e.clientY-drag.y)*.005));this._updateCameraBuffer();});
    c.addEventListener('pointerup',e=>{drag=null;c.releasePointerCapture?.(e.pointerId);});
    c.addEventListener('wheel',e=>{this.camera.radius=Math.max(0.5,Math.min(10,this.camera.radius+e.deltaY*.005));this._updateCameraBuffer();e.preventDefault();},{passive:false});
  }
  async _createPipelines(){const d=this.device,c=(l,s)=>d.createComputePipeline({label:l,layout:'auto',compute:{module:d.createShaderModule({code:s}),entryPoint:'main'}});this.pipePredict=c('FVM predict',SHADER_FVM_PREDICT);this.pipeDivergence=c('FVM divergence',SHADER_FVM_DIVERGENCE);this.pipePressure=c('FVM pressure',SHADER_FVM_PRESSURE);this.pipeCorrect=c('FVM correct',SHADER_FVM_CORRECT);this.pipeTemperature=c('FVM temperature',SHADER_FVM_TEMPERATURE);this.pipeSliceCompute=c('FVM slice',SHADER_FVM_SLICE);const vm=d.createShaderModule({code:SHADER_VERT}),fm=d.createShaderModule({code:SHADER_FRAG});this.pipeRender=d.createRenderPipeline({layout:'auto',vertex:{module:vm,entryPoint:'vs_main'},fragment:{module:fm,entryPoint:'fs_main',targets:[{format:this.format}]},primitive:{topology:'triangle-strip'}});const vv=d.createShaderModule({code:SHADER_VECTOR_VERT}),vf=d.createShaderModule({code:SHADER_VECTOR_FRAG});this.pipeVector=d.createRenderPipeline({layout:'auto',vertex:{module:vv,entryPoint:'main'},fragment:{module:vf,entryPoint:'main',targets:[{format:this.format,blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha'}}}]},primitive:{topology:'line-list'}});const gv=d.createShaderModule({code:SHADER_ROOM_VERT}),gf=d.createShaderModule({code:SHADER_ROOM_FRAG});this.pipeRoom=d.createRenderPipeline({layout:'auto',vertex:{module:gv,entryPoint:'vs_room',buffers:[{arrayStride:12,attributes:[{shaderLocation:0,offset:0,format:'float32x3'}]}]},fragment:{module:gf,entryPoint:'fs_room',targets:[{format:this.format,blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha'}}}]},primitive:{topology:'line-list'}});const sv=d.createShaderModule({code:SHADER_SLICE_VERT}),sf=d.createShaderModule({code:SHADER_SLICE_FRAG});this.pipeSliceRender=d.createRenderPipeline({layout:'auto',vertex:{module:sv,entryPoint:'main'},fragment:{module:sf,entryPoint:'main',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}});
    // 断面セル(隙間なし格子)描画パイプライン
    const scm=d.createShaderModule({code:SHADER_FVM_SLICE_CELL});
    this.pipeSliceCell=d.createRenderPipeline({label:'FVM slice cells',layout:'auto',vertex:{module:scm,entryPoint:'vs_main'},fragment:{module:scm,entryPoint:'fs_main',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}});
    // ノズル微細セルの断面スカラー描画パイプライン
    const scmN=d.createShaderModule({code:SHADER_FVM_SLICE_CELL_NOZZLE});
    this.pipeSliceCellNozzle=d.createRenderPipeline({label:'FVM slice cells (nozzle)',layout:'auto',vertex:{module:scmN,entryPoint:'vs_main'},fragment:{module:scmN,entryPoint:'fs_main',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}});
  }
  _bg(pipe,resources,label){return this.device.createBindGroup({label,layout:pipe.getBindGroupLayout(0),entries:resources.map((r,binding)=>({binding,resource:{buffer:r}}))});}
  _createBindGroups(){
    const p=this.bufParams,a=this.bufStateA,b=this.bufStateB;
    this.bgPredict=this._bg(this.pipePredict,[p,a,this.bufUStar,this.bufNozzleConnection,this.bufNeighbors,this.bufCellH],'predict');
    this.bgDivergence=this._bg(this.pipeDivergence,[p,a,this.bufUStar,this.bufDiv,this.bufNeighbors,this.bufCellH],'divergence');
    this.bgPressureAB=this._bg(this.pipePressure,[p,a,this.bufDiv,this.bufPressureA,this.bufPressureB,this.bufNeighbors,this.bufCellH],'pressure AB');
    this.bgPressureBA=this._bg(this.pipePressure,[p,a,this.bufDiv,this.bufPressureB,this.bufPressureA,this.bufNeighbors,this.bufCellH],'pressure BA');
    this.bgCorrect=this._bg(this.pipeCorrect,[p,a,this.bufUStar,this.bufPressureA,b,this.bufNeighbors,this.bufCellH],'correct');
    this.bgTemperature=this._bg(this.pipeTemperature,[p,b,a,this.bufNeighbors,this.bufCellH],'temperature');
    const r=(s,idx)=>this._bg(this.pipeRender,[p,s,this.bufCamera,idx],'render');
    this.bgRenderA=r(a,this.bufSortIdx);
    this.bgRenderB=r(b,this.bufSortIdx);
    this.bgVector=this._bg(this.pipeVector,[p,a,this.bufCamera],'vectors');
    this.bgGrid=this._bg(this.pipeRoom,[this.bufCamera],'grid');
    this.bgSliceCompute=this._bg(this.pipeSliceCompute,[p,a,this.bufCamera,this.bufSlice,this.bufGridAxes],'slice compute');
    this.bgSliceRender=this._bg(this.pipeSliceRender,[this.bufCamera,this.bufSlice,p,this.bufCamera],'slice render');
    // 断面セル描画: Params / セル状態 / カメラ / 格子面座標
    const sc=(s)=>this._bg(this.pipeSliceCell,[p,s,this.bufCamera,this.bufGridAxes],'slice cells');
    this.bgSliceCellA=sc(a);
    this.bgSliceCellB=sc(b);
    // ノズル微細セル断面用バインドグループ(gridAxesは使わない)
    const scn=(s)=>this._bg(this.pipeSliceCellNozzle,[p,s,this.bufCamera],'slice cells nozzle');
    this.bgSliceCellNozzleA=scn(a);
    this.bgSliceCellNozzleB=scn(b);
  }
  _computeStep(){const d=this.device,n=Math.ceil(this.N/256),e=d.createCommandEncoder({label:'FVM step'});let q=e.beginComputePass();q.setPipeline(this.pipePredict);q.setBindGroup(0,this.bgPredict);q.dispatchWorkgroups(n);q.end();q=e.beginComputePass();q.setPipeline(this.pipeDivergence);q.setBindGroup(0,this.bgDivergence);q.dispatchWorkgroups(n);q.end();for(let i=0;i<this.PRESSURE_ITERS;i++){q=e.beginComputePass();q.setPipeline(this.pipePressure);q.setBindGroup(0,i%2?this.bgPressureBA:this.bgPressureAB);q.dispatchWorkgroups(n);q.end();}q=e.beginComputePass();q.setPipeline(this.pipeCorrect);q.setBindGroup(0,this.bgCorrect);q.dispatchWorkgroups(n);q.end();q=e.beginComputePass();q.setPipeline(this.pipeTemperature);q.setBindGroup(0,this.bgTemperature);q.dispatchWorkgroups(n);q.end();d.queue.submit([e.finish()]);this.stepCount++;this.stats.step=this.stepCount;this.stats.time=this.stepCount*this.DT;}
  _render(){
    const v=this.context.getCurrentTexture().createView(),e=this.device.createCommandEncoder();
    if(this.visualMode===2){
      const c=e.beginComputePass();c.setPipeline(this.pipeSliceCompute);c.setBindGroup(0,this.bgSliceCompute);c.dispatchWorkgroups(9,9);c.end();
    }
    const q=e.beginRenderPass({colorAttachments:[{view:v,clearValue:{r:.05,g:.05,b:.12,a:1},loadOp:'clear',storeOp:'store'}]});
    if(this.visualMode===0){
      // 3Dメッシュ表示: 室内+ノズルの全格子ワイヤーフレーム
      q.setPipeline(this.pipeRoom);q.setBindGroup(0,this.bgGrid);q.setVertexBuffer(0,this.bufGrid);q.draw(this.gridVertexCount);
    } else if (this.visualMode===2) {
      // コンター表示(補間ベースの滑らかな断面)は解析式の滑らかなノズル外形と
      // 相性が良いため、薄いワイヤーフレーム(bufNozzleWire)を重ねて描く。
      if(this.nozzleWireVertexCount>0){
        q.setPipeline(this.pipeRoom);q.setBindGroup(0,this.bgGrid);q.setVertexBuffer(0,this.bufNozzleWire);
        q.draw(this.nozzleWireVertexCount);
      }
    } else {
      // 断面表示(visualMode===1)で速度ベクトル表示の際は、ノズル外形ワイヤーを描いて位置を明瞭にする
      if(this.displayMode===1 && this.nozzleWireVertexCount>0){
        q.setPipeline(this.pipeRoom);q.setBindGroup(0,this.bgGrid);q.setVertexBuffer(0,this.bufNozzleWire);
        q.draw(this.nozzleWireVertexCount);
      }
    }
    if(this.visualMode===2){
      if(this.displayMode!==1){
        q.setPipeline(this.pipeSliceRender);q.setBindGroup(0,this.bgSliceRender);q.draw(64*64*6);
      }
    }else if(this.visualMode===1){
      if(this.displayMode!==1){
        q.setPipeline(this.pipeSliceCell);q.setBindGroup(0,this.stateIndex?this.bgSliceCellB:this.bgSliceCellA);q.draw(6,this.NX*this.NX);
        // ノズル微細セルの断面(スカラー色分け)を室内断面の直後に重ねて描く
        if(this.nozzleCount>0){
          q.setPipeline(this.pipeSliceCellNozzle);q.setBindGroup(0,this.stateIndex?this.bgSliceCellNozzleB:this.bgSliceCellNozzleA);q.draw(6,this.nozzleCount);
        }
      }
      // ベクトルは室内格子(roomN = NX³)分だけ描画する。
      if(this.displayMode===1){q.setPipeline(this.pipeVector);q.setBindGroup(0,this.bgVector);q.draw(6,this.NX**3);}
    }else{
      if(this.displayMode!==1){
        q.setPipeline(this.pipeRender);q.setBindGroup(0,this.stateIndex?this.bgRenderB:this.bgRenderA);q.draw(4,this.N);
      }
      const roomN = this.NX ** 3;
      if(this.displayMode===1){q.setPipeline(this.pipeVector);q.setBindGroup(0,this.bgVector);q.draw(6,roomN);}
    }
    q.end();this.device.queue.submit([e.finish()]);
  }
  tick(){if(!this.paused){this._updateParamsBuffer();for(let i=0;i<this.substepsPerFrame;i++)this._computeStep();}this._render();}
  updateView(){
    if(this.visualMode!==0){
      this.camera.theta=this.sliceMode===1 ? Math.PI/2 : 0;
      this.camera.phi=Math.PI/2-0.04;
      // ── ノズル拡大 ────────────────────────────────────────────
      // 注視点を室内中心(0,0,0)からノズル軸中間高さへ寄せ、
      // カメラ半径(radius)を縮めてノズル形状とその近傍を拡大する。
      const targetY = this.nozzleFocus ? (this.DOM+this.DOM_Z)/2-this.DOM/2 : 0;
      this.camera.target=[0,targetY,0];
      if(this.nozzleFocus){
        if(this._radiusBeforeFocus===undefined) this._radiusBeforeFocus=4.0;
        this.camera.radius=0.7;
      } else if(this._radiusBeforeFocus!==undefined){
        this.camera.radius=this._radiusBeforeFocus;
        this._radiusBeforeFocus=undefined;
      }
    }else{
      this.camera.theta=.4;
      this.camera.phi=.5;
      this.camera.target=[0,0,0];
    }
    this._updateCameraBuffer();
    this._updateParamsBuffer();
  }
  setNozzleFocus(on){
    this.nozzleFocus=!!on;
    this.updateView();
  }
  setInletCondition(velocity, temp){
    if(velocity!==undefined && !isNaN(velocity)) AC.inletVelocity = velocity;
    if(temp!==undefined && !isNaN(temp)){
      AC.inletTemp = temp;
      this.T_IN = temp;
      this.stats.tMin = temp;
    }
    updateNozzleVelocities();
    this._updateParamsBuffer();
    if(this.nozzleStateData && this.device && this.bufNozzleState){
      const n=NOZZLES[this.nozzleType], core=n.innerCylinder||null, step=.18/48, roomN=this.NX**3;
      for(let i=0;i<this.nozzleCount;i++){
        const b=i*16, type=this.nozzleStateData[b+3];
        if(type===0 || type===2){
          const z=this.nozzleStateData[b+2], zp=z-this.DOM;
          const sec=getNozzleSection(this.nozzleType,zp);
          const coreR=core&&zp>=core.zStart&&zp<=core.zEnd?core.rOut:0;
          const flowArea=Math.max(Math.PI*(sec.rInner*sec.rInner-coreR*coreR),1e-8);
          const localV=Math.min(AC.inletVelocity*AC.inletArea/flowArea,35);
          this.nozzleStateData[b+6]=-localV;
          this.nozzleStateData[b+8]=this.T_IN;
          this.nozzleStateData[b+10]=localV;
        }
      }
      this.device.queue.writeBuffer(this.bufNozzleState,0,this.nozzleStateData);
      if(this.bufStateA) this.device.queue.writeBuffer(this.bufStateA,roomN*64,this.nozzleStateData);
      if(this.bufStateB) this.device.queue.writeBuffer(this.bufStateB,roomN*64,this.nozzleStateData);
    }
  }
  async setNozzle(type){this.paused=true;this.nozzleType=type;this.stepCount=0;await this.device.queue.onSubmittedWorkDone();await this._createBuffers();this._createBindGroups();this.paused=false;}
  async reset(){this.paused=true;this.stepCount=0;await this.device.queue.onSubmittedWorkDone();await this._createBuffers();this._createBindGroups();}
  destroy(){[this.bufStateA,this.bufStateB,this.bufUStar,this.bufDiv,this.bufPressureA,this.bufPressureB,this.bufParams,this.bufCamera,this.bufSortIdx,this.bufNozzleState,this.bufNozzleIdx,this.bufNozzleConnection,this.bufNeighbors,this.bufCellH,this.bufGridAxes,this.bufGrid,this.bufNozzleWire,this.bufSlice].forEach(b=>b?.destroy());this.device?.destroy?.();}
}

// ──────────────────────────────────────────────────────────────────
//  グローバル公開
//  main.js が参照している名前が分からない環境でも動くようにエイリアスを張る。
//  実体は常に統合FVM(WebGPUFVM)。LegacyWebGPUMPS は旧MPS実装(参照用)。
// ──────────────────────────────────────────────────────────────────
if (typeof window !== 'undefined') {
  window.WebGPUFVM        = WebGPUFVM;
  window.LegacyWebGPUMPS  = LegacyWebGPUMPS;
  if (!window.WebGPUMPS) window.WebGPUMPS = WebGPUFVM;
}
