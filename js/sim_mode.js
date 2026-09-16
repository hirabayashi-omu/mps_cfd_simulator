'use strict';
/**
 * sim_mode.js
 * GPU(WebGPU) / CPU(WebGL) ソルバーの手動切り替え
 *
 * ── 使い方 ─────────────────────────────────────────────────────────
 * 【A. main.js を書き換えずに使う (最小構成)】
 *   index.html で cpu_fallback.js / webgpu_mps.js の後に読み込むだけ。
 *
 *     <script src="sim_mode.js"></script>
 *
 *   画面右上にトグルが自動で表示され、切り替えると設定を localStorage に
 *   保存してページをリロードする。main.js 側が SimMode.create() を使って
 *   いなくても、下の「B」の 1 行を入れれば選択が反映される。
 *
 * 【B. main.js に組み込む (推奨)】
 *   シミュレーター生成箇所を次のように置き換える:
 *
 *     // 旧:
 *     //   let sim;
 *     //   try { sim = await new WebGPUMPS().init(canvas, nozzleType); }
 *     //   catch (e) { sim = new CPUFallbackMPS().init(canvas, nozzleType); }
 *
 *     // 新:
 *     const sim = await SimMode.create(canvas, nozzleType);
 *
 *   さらに、リロードせずその場で切り替えたい場合は再構築コールバックを登録する:
 *
 *     SimMode.onSwitch = async (mode) => {
 *       cancelAnimationFrame(rafId);
 *       sim.destroy?.();
 *       sim = await SimMode.create(canvas, sim.nozzleType);
 *       // 表示設定の引き継ぎは SimMode が自動で行う
 *       loop();
 *     };
 *
 * 【C. URL で指定】
 *   ?engine=cpu / ?engine=gpu / ?engine=auto
 *   URL 指定は localStorage より優先される。
 * ──────────────────────────────────────────────────────────────────
 */

const SimMode = {
  STORAGE_KEY: 'fluidSim.engineMode',

  /** 現在動いているシミュレーターインスタンス */
  current: null,
  /** 実際に採用されたモード ('gpu' | 'cpu') */
  activeMode: null,
  /** GPU が選べなかった場合の理由 (UI に表示) */
  fallbackReason: '',

  /**
   * その場で切り替えたい場合に main.js から登録するコールバック。
   * 未登録ならページリロードで切り替える。
   *   SimMode.onSwitch = async (mode) => { ... }
   */
  onSwitch: null,

  // ────────────────────────────────────────────────────────────────
  //  エンジンクラスの解決
  //  (プロジェクトによってクラス名が異なるため候補から探す)
  // ────────────────────────────────────────────────────────────────
  gpuClassNames: ['WebGPUFVM', 'WebGPUMPS', 'UnifiedWebGPUFVM', 'LegacyWebGPUMPS'],
  cpuClassNames: ['CPUFallbackMPS'],

  _resolve(names, override) {
    if (override) return override;
    const scope = (typeof window !== 'undefined') ? window : globalThis;
    for (const n of names) {
      // (1) window プロパティ (var/function 宣言、明示的な window.X = ... )
      if (typeof scope[n] === 'function') return scope[n];
      // (2) グローバルレキシカル宣言 (class / let / const)
      //     これらは window に載らないため、間接 eval でグローバルスコープを
      //     評価して拾う。※ type="module" で読み込んだ場合は見えないので、
      //     その場合は各ソルバー側で window.X = X を行うこと。
      try {
        const c = (0, eval)(n);
        if (typeof c === 'function') return c;
      } catch (e) { /* 未定義: 次の候補へ */ }
    }
    return null;
  },
  gpuClass: null,   // 明示指定したい場合はここに代入
  cpuClass: null,

  // ────────────────────────────────────────────────────────────────
  //  WebGPU 利用可否の判定
  // ────────────────────────────────────────────────────────────────
  async probeWebGPU() {
    if (this._probe !== undefined) return this._probe;
    let result = { ok: false, reason: '' };
    try {
      if (!navigator.gpu) {
        result.reason = 'このブラウザは WebGPU に対応していません';
      } else {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) result.reason = 'WebGPU アダプターを取得できませんでした';
        else result = { ok: true, reason: '' };
      }
    } catch (e) {
      result.reason = 'WebGPU 初期化エラー: ' + (e && e.message ? e.message : e);
    }
    this._probe = result;
    return result;
  },

  // ────────────────────────────────────────────────────────────────
  //  設定の読み書き ('auto' | 'gpu' | 'cpu')
  // ────────────────────────────────────────────────────────────────
  getPreferred() {
    try {
      const q = new URLSearchParams(location.search).get('engine');
      if (q && ['auto', 'gpu', 'cpu'].includes(q)) return q;
    } catch (e) { /* noop */ }
    try {
      const v = localStorage.getItem(this.STORAGE_KEY);
      if (v && ['auto', 'gpu', 'cpu'].includes(v)) return v;
    } catch (e) { /* noop */ }
    return 'gpu';
  },

  setPreferred(mode) {
    try { localStorage.setItem(this.STORAGE_KEY, mode); } catch (e) { /* noop */ }
  },

  // ────────────────────────────────────────────────────────────────
  //  シミュレーター生成
  //    preferred='cpu'          → CPU を強制 (WebGPU があっても使わない)
  //    preferred='gpu'          → GPU を強制 (不可なら CPU にフォールバック+警告)
  //    preferred='auto' (既定)  → GPU を試し、失敗したら CPU
  // ────────────────────────────────────────────────────────────────
  async create(canvas, nozzleType = 1, opts = {}) {
    const preferred = opts.mode || this.getPreferred();
    const GPU = this._resolve(this.gpuClassNames, this.gpuClass);
    const CPU = this._resolve(this.cpuClassNames, this.cpuClass);
    if (!CPU) {
      throw new Error(
        'CPUFallbackMPS が読み込まれていません(window.CPUFallbackMPS が未定義)。' +
        ' 考えられる原因: (1) <script src="js/cpu_fallback.js"> が sim_mode.js より前に' +
        ' 正しく読み込めているか(ブラウザのDevTools > Networkタブで404/失敗になっていないか確認)、' +
        ' (2) index.html を file:// で直接開いている場合はローカルサーバー経由' +
        '(例: python -m http.server)で開き直してみてください。'
      );
    }

    this.fallbackReason = '';

    if (preferred !== 'cpu' && GPU) {
      const probe = await this.probeWebGPU();
      if (probe.ok) {
        try {
          const sim = await new GPU().init(canvas, nozzleType);
          sim.engineMode  = sim.engineMode  || 'gpu';
          sim.engineLabel = sim.engineLabel || 'GPU (WebGPU)';
          this.current = sim;
          this.activeMode = 'gpu';
          this._refreshToggle();
          return sim;
        } catch (e) {
          this.fallbackReason = 'GPU 初期化に失敗したため CPU に切り替えました: ' +
            (e && e.message ? e.message : e);
          console.warn('[SimMode]', this.fallbackReason);
        }
      } else {
        this.fallbackReason = probe.reason;
        if (preferred === 'gpu') console.warn('[SimMode]', probe.reason);
      }
    } else if (preferred !== 'cpu' && !GPU) {
      this.fallbackReason = 'GPU ソルバー(webgpu_mps.js)が読み込まれていません';
    }

    const sim = await new CPU().init(canvas, nozzleType);
    this.current = sim;
    this.activeMode = 'cpu';
    this._refreshToggle();
    return sim;
  },

  // ────────────────────────────────────────────────────────────────
  //  手動切り替え
  //   onSwitch が登録されていればそれを呼ぶ。無ければリロード。
  // ────────────────────────────────────────────────────────────────
  async switchTo(mode) {
    if (mode === this.activeMode && this.getPreferred() === mode) return;
    this.setPreferred(mode);

    if (typeof this.onSwitch === 'function') {
      const carry = this.captureState(this.current);
      await this.onSwitch(mode, carry);
      this.restoreState(this.current, carry);
      this._refreshToggle();
      return;
    }

    // コールバック未登録: URL に反映してリロード (main.js の書き換え不要)
    try {
      const url = new URL(location.href);
      url.searchParams.set('engine', mode);
      location.href = url.toString();
    } catch (e) {
      location.reload();
    }
  },

  // ── 切り替え時に引き継ぐ表示・計算設定 ────────────────────────────
  captureState(sim) {
    if (!sim) return null;
    return {
      nozzleType   : sim.nozzleType,
      displayMode  : sim.displayMode,
      visualMode   : sim.visualMode,
      sliceMode    : sim.sliceMode,
      slicePosition: sim.slicePosition,
      nozzleFocus  : sim.nozzleFocus,
      paused       : sim.paused,
      inletVelocity: (typeof AC !== 'undefined') ? AC.inletVelocity : undefined,
      inletTemp    : (typeof AC !== 'undefined') ? AC.inletTemp     : undefined,
    };
  },

  restoreState(sim, st) {
    if (!sim || !st) return;
    sim.displayMode   = st.displayMode;
    sim.visualMode    = st.visualMode;
    sim.sliceMode     = st.sliceMode;
    sim.slicePosition = st.slicePosition;
    if (st.inletVelocity !== undefined && sim.setInletCondition) {
      sim.setInletCondition(st.inletVelocity, st.inletTemp);
    }
    sim.updateView?.();
    if (st.nozzleFocus !== undefined) sim.setNozzleFocus?.(st.nozzleFocus);
    sim.paused = st.paused;
  },

  // ────────────────────────────────────────────────────────────────
  //  UI トグル
  // ────────────────────────────────────────────────────────────────
  async installToggle(options = {}) {
    if (this._ui) return this._ui;
    const host = options.container || document.body;

    const wrap = document.createElement('div');
    wrap.id = 'sim-mode-toggle';
    wrap.style.cssText = [
      'position:absolute', 'top:12px', 'right:12px', 'z-index:50',
      'font:12px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif',
      'color:#dfe6f2', 'background:rgba(16,20,32,.82)',
      'border:1px solid rgba(120,150,200,.35)', 'border-radius:8px',
      'padding:8px 10px', 'backdrop-filter:blur(4px)',
      'user-select:none', 'min-width:150px',
    ].join(';');

    const title = document.createElement('div');
    title.textContent = 'ソルバー';
    title.style.cssText = 'opacity:.65;margin-bottom:5px;letter-spacing:.06em';

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:4px';

    const mkBtn = (mode, label) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.dataset.mode = mode;
      b.textContent = label;
      b.style.cssText = [
        'flex:1', 'cursor:pointer', 'padding:5px 6px', 'border-radius:5px',
        'border:1px solid rgba(120,150,200,.35)', 'background:transparent',
        'color:inherit', 'font:inherit', 'transition:background .15s',
      ].join(';');
      b.addEventListener('click', () => this.switchTo(mode));
      return b;
    };

    const btnGpu = mkBtn('gpu', 'GPU');
    const btnCpu = mkBtn('cpu', 'CPU');
    row.append(btnGpu, btnCpu);

    const note = document.createElement('div');
    note.style.cssText = 'margin-top:5px;opacity:.6;font-size:11px;line-height:1.35';

    wrap.append(title, row, note);
    if (getComputedStyle(host).position === 'static' && host !== document.body) {
      host.style.position = 'relative';
    }
    host.appendChild(wrap);

    this._ui = { wrap, btnGpu, btnCpu, note };

    const probe = await this.probeWebGPU();
    if (!probe.ok) {
      btnGpu.disabled = true;
      btnGpu.style.opacity = '.4';
      btnGpu.style.cursor = 'not-allowed';
      btnGpu.title = probe.reason;
    }
    this._refreshToggle();
    return this._ui;
  },

  _refreshToggle() {
    const ui = this._ui;
    if (!ui) return;
    const active = this.activeMode;
    for (const b of [ui.btnGpu, ui.btnCpu]) {
      const on = b.dataset.mode === active;
      b.style.background = on ? 'rgba(90,140,230,.42)' : 'transparent';
      b.style.borderColor = on ? 'rgba(130,180,255,.8)' : 'rgba(120,150,200,.35)';
      b.style.fontWeight = on ? '600' : '400';
    }
    const pref = this.getPreferred();
    if (this.fallbackReason && active === 'cpu' && pref !== 'cpu') {
      ui.note.textContent = this.fallbackReason;
    } else if (active) {
      ui.note.textContent = (active === 'gpu')
        ? 'WebGPU / 統合FVM'
        : 'WebGL / 非均一格子FVM';
    } else {
      ui.note.textContent = '';
    }
  },
};

if (typeof window !== 'undefined') {
  if (location.protocol === 'file:') {
    console.warn(
      '[SimMode] このページは file:// で開かれています。WebGPU は https/localhost などの' +
      'セキュアコンテキストでしか動作せず、file:// ではブラウザによりスクリプト読み込みが' +
      '不安定になることもあります。ローカルサーバー(例: python -m http.server)経由で' +
      '開くことを推奨します。'
    );
  }
  window.SimMode = SimMode;
  // トグルは自動設置 (不要なら SimMode.autoInstall = false を先に設定)
  window.addEventListener('DOMContentLoaded', () => {
    if (SimMode.autoInstall === false) return;
    const host = document.querySelector('#canvas-wrap, #viewport, .canvas-container') || document.body;
    SimMode.installToggle({ container: host });
  });
}
