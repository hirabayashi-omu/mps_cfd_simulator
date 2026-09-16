'use strict';
/**
 * main.js
 * アプリケーションコントローラー
 * - WebGPU / CPUフォールバック自動選択
 * - UI イベント処理
 * - アニメーションループ
 * - 統計表示
 */

// ─────────────────────────────────────────────────────────────────
//  グローバル状態
// ─────────────────────────────────────────────────────────────────
let sim        = null;   // WebGPUMPS | CPUFallbackMPS
let useWebGPU  = false;
let animId     = null;
let selectedNozzle = 1;
let fpsHistory = [];
let lastTime   = performance.now();

// ─────────────────────────────────────────────────────────────────
//  DOM 要素
// ─────────────────────────────────────────────────────────────────
let canvas        = document.getElementById('sim-canvas');
const btnStart    = document.getElementById('btn-start');
const btnPause    = document.getElementById('btn-pause');
const btnReset    = document.getElementById('btn-reset');
const nozzleCards = document.querySelectorAll('.nozzle-card');
const modeButtons = document.querySelectorAll('.mode-btn');
const viewButtons = document.querySelectorAll('.view-btn');
const statusBadge = document.getElementById('status-badge');
const gpuBadge    = document.getElementById('gpu-badge');
const elStep      = document.getElementById('stat-step');
const elTime      = document.getElementById('stat-time');
const elTmin      = document.getElementById('stat-tmin');
const elTmax      = document.getElementById('stat-tmax');
const elVmax      = document.getElementById('stat-vmax');
const elFps       = document.getElementById('stat-fps');
const elParticles = document.getElementById('stat-particles');
const elProgress  = document.getElementById('progress-bar');
const elSubstep   = document.getElementById('substep-slider');
const elSubstepVal= document.getElementById('substep-value');
const colorbarCanvas = document.getElementById('colorbar-canvas');
const overlay     = document.getElementById('loading-overlay');
const overlayMsg  = document.getElementById('loading-msg');
const nozzleInfo  = document.getElementById('nozzle-info');
const sliceAxis   = document.getElementById('slice-axis');
const slicePosition = document.getElementById('slice-position');
const slicePositionValue = document.getElementById('slice-position-value');
const sliceControls = document.getElementById('slice-controls');
const btnNozzleZoom = document.getElementById('btn-nozzle-zoom');
const elInletVel    = document.getElementById('inlet-vel-slider');
const elInletVelVal = document.getElementById('inlet-vel-val');
const elInletTemp   = document.getElementById('inlet-temp-slider');
const elInletTempVal= document.getElementById('inlet-temp-val');
const elInletPress  = document.getElementById('inlet-press-slider');
const elInletPressVal= document.getElementById('inlet-press-val');
const btnInletReset = document.getElementById('btn-inlet-reset');
const elCondVel     = document.getElementById('cond-inlet-vel');
const elCondTemp    = document.getElementById('cond-inlet-temp');
const btnTheory     = document.getElementById('btn-theory');
const modalTheory   = document.getElementById('theory-modal');
const modalClose    = document.getElementById('modal-close');
const modalTabs     = document.querySelectorAll('.modal-tab');
const modalPanes    = document.querySelectorAll('.modal-pane');

// ─────────────────────────────────────────────────────────────────
//  GPU/CPU 手動切替 (sim_mode.js)
//  - sim_mode.js のデフォルトの自動設置(DOMContentLoaded時)は使わず、
//    startApp() 内でヘッダーの gpu-badge の隣に明示的に設置する。
//    autoInstall はここ(main.js の先頭、同期実行部分)で下げておく必要がある。
//    main.js は他の <script> タグと同様パース順に同期実行されるため、
//    後で発火する DOMContentLoaded より確実に先に実行される。
//  - sim_mode.js が読み込まれていない環境でも動くよう typeof チェックする。
// ─────────────────────────────────────────────────────────────────
if (typeof SimMode !== 'undefined') SimMode.autoInstall = false;

// WebGPU と WebGL は同一 <canvas> 要素上で共存できない
// (一度 getContext() したコンテキスト種別は同じ要素に固定される)ため、
// GPU⇔CPU をその場で切り替える際は canvas 要素そのものを新しいものに
// 差し替える。ID・クラス・現在の描画サイズは引き継ぐ。
function replaceCanvas() {
  const fresh = document.createElement('canvas');
  fresh.id = canvas.id;
  fresh.className = canvas.className;
  fresh.width  = canvas.width;
  fresh.height = canvas.height;
  canvas.replaceWith(fresh);
  canvas = fresh;
  return canvas;
}

// gpu-badge の表示をエンジン切替の実態(SimMode.activeMode)に合わせる。
function updateGpuBadge() {
  const gpuActive = (typeof SimMode !== 'undefined' && SimMode.activeMode)
    ? SimMode.activeMode === 'gpu'
    : useWebGPU;
  if (gpuActive) {
    gpuBadge.textContent = '⚡ WebGPU';
    gpuBadge.className   = 'badge badge-gpu';
  } else {
    gpuBadge.textContent = '🔄 CPU (WebGL)';
    gpuBadge.className   = 'badge badge-cpu';
  }
  if (typeof SimMode !== 'undefined' && SimMode.fallbackReason && !gpuActive) {
    gpuBadge.title = SimMode.fallbackReason;
  } else {
    gpuBadge.removeAttribute('title');
  }
}

// ─────────────────────────────────────────────────────────────────
//  サイドバー折りたたみ & タブ切り替え
// ─────────────────────────────────────────────────────────────────
function setupSidebarUI() {
  const elLeftPanel   = document.getElementById('left-panel');
  const btnToggleSide = document.getElementById('btn-toggle-sidebar');
  const btnCloseSide  = document.getElementById('btn-close-sidebar');
  const btnOpenSide   = document.getElementById('btn-open-sidebar');

  function toggleSidebar(forceState) {
    if (!elLeftPanel) return;
    const isCollapsed = (forceState !== undefined) ? forceState : !elLeftPanel.classList.contains('collapsed');
    elLeftPanel.classList.toggle('collapsed', isCollapsed);
    if (btnOpenSide) btnOpenSide.classList.toggle('visible', isCollapsed);
  }

  btnToggleSide?.addEventListener('click', () => toggleSidebar());
  btnCloseSide?.addEventListener('click', () => toggleSidebar(true));
  btnOpenSide?.addEventListener('click', () => toggleSidebar(false));

  const tabBtns  = document.querySelectorAll('.sidebar-tabs .tab-btn');
  const tabPanes = document.querySelectorAll('.sidebar-tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.dataset.tab;
      tabBtns.forEach(b => b.classList.toggle('active', b === btn));
      tabPanes.forEach(pane => {
        pane.classList.toggle('active', pane.id === targetId);
      });
    });
  });
}

async function startApp() {
  setupSidebarUI();
  showLoading('システムを初期化中...');

  // ノズルSVGプレビュー生成
  for (let n = 1; n <= 4; n++) {
    const el = document.getElementById(`nozzle-svg-${n}`);
    if (el) el.innerHTML = nozzleSVG(n);
  }

  // カラーバー描画
  drawColorbar();

  if (typeof SimMode !== 'undefined') {
    // ── sim_mode.js あり: 保存済みの選択(既定 auto)に従ってソルバーを生成 ──
    //   auto: WebGPU対応なら GPU、非対応/初期化失敗時は CPU に自動フォールバック。
    //   gpu / cpu: ユーザーがトグルで明示的に選んだモードを次回起動時も維持する。
    showLoading('ソルバーを初期化中...');
    sim = await SimMode.create(canvas, selectedNozzle);
    useWebGPU = (SimMode.activeMode === 'gpu');
    updateGpuBadge();
    elParticles.textContent = sim.N.toLocaleString();

    // 右上のGPU/CPU切替トグルをヘッダーの gpu-badge の隣に設置する
    const toggleUi = await SimMode.installToggle({ container: document.querySelector('.header-actions') });
    if (toggleUi?.wrap) {
      // installToggle は独立パネルとして「ビューポートに重ねる浮動表示」を
      // 前提に position:absolute で作る。ここではヘッダーのボタン列に
      // インライン表示したいので配置だけ上書きし、gpu-badge の直後に移動する。
      toggleUi.wrap.style.position = 'static';
      toggleUi.wrap.style.minWidth = 'auto';
      toggleUi.wrap.style.display  = 'inline-flex';
      toggleUi.wrap.style.alignItems = 'center';
      toggleUi.wrap.style.gap      = '4px';
      toggleUi.wrap.style.padding  = '3px 5px';
      toggleUi.wrap.firstElementChild?.remove(); // 「ソルバー」ラベル行は省略(gpu-badgeと重複するため)
      toggleUi.note.style.display  = 'none';      // 補足テキストも省略(gpu-badgeのtitleに集約)
      gpuBadge.insertAdjacentElement('afterend', toggleUi.wrap);
    }

    // ── その場切替 (ページリロードなし) ──────────────────────────────
    // carry には切替前の displayMode / visualMode 等が入っている
    // (SimMode.switchTo が captureState() で退避したもの)。
    // SimMode 側がこれを sim.* へ再適用するのは onSwitch から戻った後なので、
    // ここで参照するUI同期(カラーバー・ボタンのactive)には carry を使う。
    SimMode.onSwitch = async (mode, carry) => {
      if (animId) cancelAnimationFrame(animId);
      showLoading(mode === 'gpu'
        ? 'WebGPU (統合FVM) を初期化中...'
        : 'CPU (WebGL) フォールバックへ切替中...');
      try {
        sim?.destroy?.();
        replaceCanvas(); // WebGPU/WebGL はコンテキスト種別を跨げないためcanvasを差し替える
        sim = await SimMode.create(canvas, selectedNozzle, { mode });
        useWebGPU = (SimMode.activeMode === 'gpu');
        updateGpuBadge();
        elParticles.textContent = sim.N.toLocaleString();
        updateNozzleInfo();
        updateColorbarLabels(carry?.displayMode ?? sim.displayMode);
        if (carry) {
          modeButtons.forEach(b => b.classList.toggle('active', parseInt(b.dataset.mode, 10) === carry.displayMode));
          viewButtons.forEach(b => b.classList.toggle('active', parseInt(b.dataset.view, 10) === carry.visualMode));
          sliceControls?.classList.toggle('visible', carry.visualMode !== 0);
          nozzleZoom = !!carry.nozzleFocus;
          applyNozzleZoomButtonStyle();
        }
        // 切替前の一時停止状態を引き継ぐ(SimMode.restoreState が直後に
        // sim.paused へ同じ値を再セットするが、ボタン/バッジの見た目も
        // ここで合わせておく)
        const wasPaused = carry?.paused ?? false;
        sim.paused = wasPaused;
        updateButtonState(!wasPaused);
        statusBadge.textContent = wasPaused ? '一時停止' : '解析中';
        statusBadge.className   = wasPaused ? 'status-badge paused' : 'status-badge running';
        startLoop();
      } catch (e) {
        console.error('[CFD] ソルバー切替に失敗:', e);
        statusBadge.textContent = 'エラー';
        statusBadge.className   = 'status-badge stopped';
      } finally {
        hideLoading();
      }
    };
  } else {
    // ── sim_mode.js なし: 従来通りの自動判定 (後方互換) ──────────────
    useWebGPU = !!navigator.gpu;
    try {
      if (useWebGPU) {
        showLoading('WebGPU を初期化中... (FVM 構造格子)');
        sim = new WebGPUMPS();
        await sim.init(canvas, selectedNozzle);
      } else {
        throw new Error('WebGPU 非対応');
      }
    } catch (e) {
      console.warn('WebGPU 初期化失敗:', e.message, '→ CPUフォールバックに切り替え');
      useWebGPU = false;
      showLoading('CPU フォールバックモードで起動中...');
      sim = new CPUFallbackMPS();
      sim.init(canvas, selectedNozzle);
    }
    updateGpuBadge();
    elParticles.textContent = sim.N.toLocaleString();
  }

  hideLoading();
  updateNozzleInfo();
  updateColorbarLabels(sim.displayMode);
  const vpLabel = document.getElementById('vp-mode-label');
  if (vpLabel) {
    const labels = ['表示: 温度分布 [°C]', '表示: 速度場 [m/s]', '表示: 圧力場 [Pa]'];
    vpLabel.textContent = labels[sim.displayMode] || '';
  }
  modeButtons.forEach(b => b.classList.toggle('active', parseInt(b.dataset.mode, 10) === sim.displayMode));
  viewButtons.forEach(b => b.classList.toggle('active', parseInt(b.dataset.view, 10) === sim.visualMode));
  sliceControls?.classList.toggle('visible', sim.visualMode !== 0);
  if (slicePosition) slicePosition.value = String(sim.slicePosition ?? 1);
  if (slicePositionValue) slicePositionValue.textContent = (sim.slicePosition ?? 1).toFixed(2) + ' m';
  if (sliceAxis) sliceAxis.value = String(sim.sliceMode ?? 1);
  sim.setNozzleFocus?.(nozzleZoom);
  applyNozzleZoomButtonStyle();
  sim.updateView?.();

  // 初期状態で停止したままだと、キャンバスが静止画に見えるため自動開始する。
  sim.paused = false;
  updateButtonState(true);
  statusBadge.textContent = '解析中';
  statusBadge.className   = 'status-badge running';
  startLoop();
}

// ─────────────────────────────────────────────────────────────────
//  アニメーションループ
// ─────────────────────────────────────────────────────────────────
function startLoop() {
  function loop() {
    animId = requestAnimationFrame(loop);

    // FPS 計算
    const now = performance.now();
    const dt  = now - lastTime;
    lastTime  = now;
    fpsHistory.push(1000 / dt);
    if (fpsHistory.length > 30) fpsHistory.shift();
    const fps = fpsHistory.reduce((a,b)=>a+b,0) / fpsHistory.length;

    sim.tick();
    if (++statFrameCount % 3 === 0) {
      updateStats(fps);
    }
  }
  loop();
}

// ─────────────────────────────────────────────────────────────────
//  統計表示更新
// ─────────────────────────────────────────────────────────────────
function updateStats(fps) {
  const s = sim.stats;
  elStep.textContent = s.step.toLocaleString();
  elTime.textContent = s.time.toFixed(2) + ' s';
  elFps.textContent  = fps.toFixed(1);

  if (s.tMin && s.tMax) {
    elTmin.textContent = s.tMin.toFixed(1) + ' °C';
    elTmax.textContent = s.tMax.toFixed(1) + ' °C';
    elVmax.textContent = (s.vMax || 0).toFixed(1) + ' m/s';
  }

  // プログレスバー (100ステップで100%)
  const prog = Math.min((sim.stepCount % 200) / 200 * 100, 100);
  elProgress.style.width = prog + '%';

  if (sim.steadyComplete) {
    statusBadge.textContent = '疑似定常計算完了';
    statusBadge.className = 'status-badge stopped';
    btnStart.disabled = false;
    btnPause.disabled = true;
  }
}

// ─────────────────────────────────────────────────────────────────
//  UI ボタンイベント
// ─────────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  if (!sim) return;
  sim.paused = false;
  sim.steadyComplete = false;
  updateButtonState(true);
  statusBadge.textContent = '解析中';
  statusBadge.className   = 'status-badge running';
});

btnPause.addEventListener('click', () => {
  if (!sim) return;
  sim.paused = !sim.paused;
  btnPause.textContent = sim.paused ? '▶ 再開' : '⏸ 一時停止';
  statusBadge.textContent = sim.paused ? '一時停止' : '解析中';
  statusBadge.className   = sim.paused ? 'status-badge paused' : 'status-badge running';
});

btnReset.addEventListener('click', async () => {
  if (!sim) return;
  showLoading('リセット中...');
  await sim.reset();
  sim.paused = true;
  updateButtonState(false);
  statusBadge.textContent = '停止';
  statusBadge.className   = 'status-badge stopped';
  hideLoading();
});

// ─────────────────────────────────────────────────────────────────
//  ノズル選択
// ─────────────────────────────────────────────────────────────────
nozzleCards.forEach(card => {
  card.addEventListener('click', async () => {
    const type = parseInt(card.dataset.nozzle);
    if (type === selectedNozzle) return;

    selectedNozzle = type;
    nozzleCards.forEach(c => c.classList.remove('active'));
    card.classList.add('active');

    showLoading(`ノズル ${NOZZLES[type].name} に切り替え中...`);
    await sim.setNozzle(type);
    updateNozzleInfo();
    elParticles.textContent = sim.N.toLocaleString();
    hideLoading();
  });
});

// ─────────────────────────────────────────────────────────────────
//  表示モード切替
// ─────────────────────────────────────────────────────────────────
modeButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const mode = parseInt(btn.dataset.mode);
    sim.displayMode = mode;
    if (useWebGPU) sim._updateParamsBuffer();
    modeButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    updateColorbarLabels(mode);
    const vpLabel = document.getElementById('vp-mode-label');
    if (vpLabel) {
      const labels = ['表示: 温度分布 [°C]', '表示: 速度場 [m/s]', '表示: 圧力場 [Pa]'];
      vpLabel.textContent = labels[mode] || '';
    }
  });
});

// 表示形式切替: 0=3D粒子, 1=断面, 2=コンター
viewButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const view = parseInt(btn.dataset.view);
    if (!sim) return;
    sim.visualMode = view;
    viewButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    sliceControls?.classList.toggle('visible', view !== 0);
    sim.updateView?.();
  });
});

function updateSlice() {
  if (!sim) return;
  const axis = parseInt(sliceAxis.value);
  const position = parseFloat(slicePosition.value);
  slicePositionValue.textContent = position.toFixed(2) + ' m';
  sim.sliceMode = axis;
  sim.slicePosition = position;
  sim.updateView?.();
}

sliceAxis?.addEventListener('change', updateSlice);
slicePosition?.addEventListener('input', updateSlice);

// ─────────────────────────────────────────────────────────────────
//  ノズル部拡大トグル (断面・コンター表示時のみ有効)
//  ノズル(天井中央の吹出し口)とその近傍だけを大きく見たいという要望に
//  対応。カメラの注視点をノズル軸付近(天井やや上)に移し、距離を縮める。
//  ON/OFF は sim.setNozzleFocus() (CPU/GPU共通インターフェース) 経由。
// ─────────────────────────────────────────────────────────────────
let nozzleZoom = true;

function applyNozzleZoomButtonStyle() {
  if (!btnNozzleZoom) return;
  btnNozzleZoom.classList.toggle('active', nozzleZoom);
  btnNozzleZoom.style.background = nozzleZoom ? 'rgba(79,195,247,0.35)' : 'rgba(79,195,247,0.10)';
  btnNozzleZoom.style.color      = nozzleZoom ? '#e8f7ff' : 'var(--text-secondary)';
  btnNozzleZoom.style.borderColor = nozzleZoom ? 'rgba(79,195,247,0.9)' : 'rgba(79,195,247,0.35)';
}

btnNozzleZoom?.addEventListener('click', () => {
  nozzleZoom = !nozzleZoom;
  applyNozzleZoomButtonStyle();
  sim?.setNozzleFocus?.(nozzleZoom);
});

// ─────────────────────────────────────────────────────────────────
//  サブステップスライダー
// ─────────────────────────────────────────────────────────────────
if (elSubstep) {
  elSubstep.addEventListener('input', () => {
    const v = parseInt(elSubstep.value);
    sim.substepsPerFrame = v;
    elSubstepVal.textContent = v;
  });
}

// ─────────────────────────────────────────────────────────────────
//  ノズル上面（入口面）条件スライダー
// ─────────────────────────────────────────────────────────────────
let inletRafPending = false;
function updateInletCondition() {
  if (!elInletVel || !elInletTemp) return;
  const vel = parseFloat(elInletVel.value);
  const temp = parseFloat(elInletTemp.value);
  const press = elInletPress ? parseFloat(elInletPress.value) : 60.0;
  if (elInletVelVal) elInletVelVal.textContent = vel.toFixed(1) + ' m/s';
  if (elInletTempVal) elInletTempVal.textContent = temp.toFixed(1) + ' °C';
  if (elInletPressVal) elInletPressVal.textContent = Math.round(press) + ' Pa';
  if (elCondVel) elCondVel.textContent = vel.toFixed(1) + ' m/s';
  if (elCondTemp) elCondTemp.textContent = temp.toFixed(1) + ' °C';

  if (!inletRafPending) {
    inletRafPending = true;
    requestAnimationFrame(() => {
      inletRafPending = false;
      if (sim && typeof sim.setInletCondition === 'function') {
        sim.setInletCondition(vel, temp, press);
      }
      updateNozzleInfo();
      updateColorbarLabels(sim ? sim.displayMode : 0);
    });
  }
}

if (elInletVel) {
  elInletVel.addEventListener('input', updateInletCondition);
}
if (elInletTemp) {
  elInletTemp.addEventListener('input', updateInletCondition);
}
if (elInletPress) {
  elInletPress.addEventListener('input', updateInletCondition);
}
if (btnInletReset) {
  btnInletReset.addEventListener('click', () => {
    if (elInletVel) elInletVel.value = '9.6';
    if (elInletTemp) elInletTemp.value = '23.5';
    if (elInletPress) elInletPress.value = '60';
    updateInletCondition();
  });
}

// ─────────────────────────────────────────────────────────────────
//  カラーバー描画
// ─────────────────────────────────────────────────────────────────
function jetColorFallback(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if      (t < 0.125) { r=0;         g=0;              b=0.5+4*t; }
  else if (t < 0.375) { r=0;         g=4*(t-0.125);    b=1; }
  else if (t < 0.625) { r=4*(t-0.375); g=1;            b=1-4*(t-0.375); }
  else if (t < 0.875) { r=1;         g=1-4*(t-0.625);  b=0; }
  else                { r=1-4*(t-0.875); g=0;           b=0; }
  return [r, g, b];
}

function drawColorbar() {
  const getColor = (typeof jetColorJS === 'function') ? jetColorJS : jetColorFallback;
  if (!colorbarCanvas) return;
  const ctx = colorbarCanvas.getContext('2d');
  const w = colorbarCanvas.width = 24;
  const h = colorbarCanvas.height = 200;

  const grad = ctx.createLinearGradient(0, h, 0, 0);
  for (let i = 0; i <= 20; i++) {
    const t  = i / 20;
    const [r,g,b] = getColor(t);
    grad.addColorStop(t, `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`);
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
}

function updateColorbarLabels(mode) {
  const minEl = document.getElementById('colorbar-min');
  const maxEl = document.getElementById('colorbar-max');
  const titleEl = document.getElementById('colorbar-title');
  if (!minEl) return;
  if (mode === 0) {
    titleEl.textContent = '温度 [°C]';
    minEl.textContent   = (AC.inletTemp || 23.5).toFixed(1);
    maxEl.textContent   = '34.0';
  } else if (mode === 1) {
    titleEl.textContent = '速度 [m/s]';
    minEl.textContent   = '0';
    maxEl.textContent   = '20+';
  } else {
    titleEl.textContent = '圧力 [Pa]';
    minEl.textContent   = '0';
    const pMax = sim ? (sim.inletPressure ?? 60) : 60;
    maxEl.textContent   = Math.round(pMax) + '+';
  }
}

// ─────────────────────────────────────────────────────────────────
//  ノズル情報パネル
// ─────────────────────────────────────────────────────────────────
function updateNozzleInfo() {
  const n = NOZZLES[selectedNozzle];
  if (!nozzleInfo) return;
  nozzleInfo.innerHTML = `
    <div class="info-row"><span>名称</span><span>${n.name} ${n.label}</span></div>
    <div class="info-row"><span>出口内径</span><span>Ø${(n.outlet.innerR*2000).toFixed(0)} mm</span></div>
    <div class="info-row"><span>出口面積</span><span>${(n.outlet.area*1e6).toFixed(1)} cm²</span></div>
    <div class="info-row"><span>出口風速</span><span>${n.outlet.velocity.toFixed(1)} m/s</span></div>
    <div class="info-row"><span>流量</span><span>${(AC.massFlow*1000).toFixed(1)} g/s</span></div>
    <div class="info-row desc"><span>${n.desc}</span></div>
  `;
}

// ─────────────────────────────────────────────────────────────────
//  ローディング表示
// ─────────────────────────────────────────────────────────────────
function showLoading(msg) {
  if (!overlay) return;
  overlayMsg.textContent = msg || '処理中...';
  overlay.classList.remove('hidden');
}
function hideLoading() {
  if (!overlay) return;
  overlay.classList.add('hidden');
}

// ─────────────────────────────────────────────────────────────────
//  ボタン状態更新
// ─────────────────────────────────────────────────────────────────
function updateButtonState(running) {
  btnStart.disabled  = running;
  btnPause.disabled  = !running;
  btnPause.textContent = '⏸ 一時停止';
}

// ─────────────────────────────────────────────────────────────────
//  基礎原理モーダル制御
// ─────────────────────────────────────────────────────────────────
function openTheoryModal() {
  if (!modalTheory) return;
  modalTheory.classList.add('open');
}

function closeTheoryModal() {
  if (!modalTheory) return;
  modalTheory.classList.remove('open');
}

if (btnTheory) {
  btnTheory.addEventListener('click', openTheoryModal);
}
if (modalClose) {
  modalClose.addEventListener('click', closeTheoryModal);
}
if (modalTheory) {
  modalTheory.addEventListener('click', (e) => {
    if (e.target === modalTheory) closeTheoryModal();
  });
}
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeTheoryModal();
});

// モーダル内タブ切り替え
if (modalTabs.length) {
  modalTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      modalTabs.forEach(t => t.classList.remove('active'));
      modalPanes.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const pane = document.getElementById(`pane-${target}`);
      if (pane) pane.classList.add('active');
    });
  });
}


// ─────────────────────────────────────────────────────────────────
//  k-ε 乱流モデル UI トグル連携
// ─────────────────────────────────────────────────────────────────
const btnTurb       = document.getElementById('btn-turb');
const btnTurbPanel  = document.getElementById('btn-turb-panel');
const turbStatusLbl = document.getElementById('turb-status-label');

function updateTurbUI(on) {
  const text      = on ? '🌀 k-ε 乱流: ON' : '🌀 k-ε 乱流: OFF';
  const panelText = on ? '🌀 k-ε 乱流 (ON)' : '🌀 k-ε 乱流 (OFF)';
  const labelText = on ? '有効' : '無効';
  const color     = on ? '#00e5ff' : '#888';
  const bg        = on ? 'rgba(0,229,255,0.15)' : 'rgba(255,255,255,0.05)';

  if (btnTurb) {
    btnTurb.textContent = text;
    btnTurb.style.borderColor = color;
    btnTurb.style.color = color;
    btnTurb.style.background = bg;
  }
  if (btnTurbPanel) {
    btnTurbPanel.textContent = panelText;
    btnTurbPanel.style.borderColor = color;
    btnTurbPanel.style.color = color;
    btnTurbPanel.style.background = bg;
  }
  if (turbStatusLbl) {
    turbStatusLbl.textContent = labelText;
    turbStatusLbl.style.color = color;
  }
}

function toggleTurbulence() {
  if (!sim) return;
  const next = !sim.useTurbulence;
  sim.useTurbulence = next;
  if (typeof sim.toggleTurbulence === 'function') {
    sim.toggleTurbulence(next);
  }
  updateTurbUI(next);
}

if (btnTurb)      btnTurb.addEventListener('click', toggleTurbulence);
if (btnTurbPanel) btnTurbPanel.addEventListener('click', toggleTurbulence);

// ─────────────────────────────────────────────────────────────────
//  起動
// ─────────────────────────────────────────────────────────────────
window.addEventListener('load', startApp);
window.addEventListener('beforeunload', () => {
  if (animId) cancelAnimationFrame(animId);
  sim?.destroy();
});
