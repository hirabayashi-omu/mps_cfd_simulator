'use strict';
/**
 * nozzle.js
 * 4種ノズルの形状定義と流体力学パラメータ
 *
 * 座標系: z=0が部屋底面(床)、z=2.0mが天井
 * ノズルは天井(z=2.0m)の中央穴に接続、下部テーパーが室内に向く
 * ノズル高さ: 180mm (上部テーパー140mm + 下部ストレート40mm)
 */

// ─────────────────────────────────────────────────────────────────
//  AC仕様
// ─────────────────────────────────────────────────────────────────
const AC = {
  inletDiam   : 0.110,  // m  (吹出し口径 Ø110mm)
  inletRadius  : 0.055, // m  (上部入口半径)
  inletVelocity: 9.6,   // m/s (吹出し口風速)
  inletTemp   : 23.5,   // °C
  ambientTemp : 34.0,   // °C
  rho         : 1.2,    // kg/m³ (空気密度)
};
AC.inletArea = Math.PI * (AC.inletDiam / 2) ** 2;
AC.massFlow  = AC.rho * AC.inletVelocity * AC.inletArea;  // kg/s

// ─────────────────────────────────────────────────────────────────
//  ノズル定義 (4種)
//  profile: [[z_from_bottom(m), r_inner(m), r_outer(m)], ...]
//           z=0: 下端(室内側出口), z=0.18: 上端(AC接続側)
// ─────────────────────────────────────────────────────────────────
const NOZZLES = {
  1: {
    id   : 1,
    name : '図1',
    label: '標準テーパー (上部整流円筒付)',
    desc : '喉部から上部に内径Ø40/外径Ø48の整流円筒を配置',
    color: '#4FC3F7',
    colorHex: 0x4FC3F7,
    profile: [
      [0.000, 0.0425, 0.0465],   // 下端(出口): Ø85 / Ø93
      [0.040, 0.0200, 0.0240],   // 喉部       : Ø40 / Ø48
      [0.180, 0.0550, 0.0590],   // 上端(入口) : Ø110 / Ø118
    ],
    innerStructure: {
      type: 'rectifying_cylinder',
      rIn : 0.0200,   // Ø40mm / 2
      rOut: 0.0240,   // Ø48mm / 2
      zStart: 0.040,
      zEnd  : 0.075,  // スロートから約35mm上方に伸びる
    },
    outlet: {
      innerR  : 0.0425,          // Ø85mm/2
      type    : 'uniform',
      hasCore : false,
      coreR   : 0,
    },
  },

  2: {
    id   : 2,
    name : '図2',
    label: '中空コーン＋円筒 二重管型',
    desc : '上部に中空コーン整流筒(Ø85/Ø93→Ø15/Ø23)、下部に円筒整流筒(Ø15/Ø23)を持つ二重管',
    color: '#81C784',
    colorHex: 0x81C784,
    profile: [
      [0.000, 0.0425, 0.0465],
      [0.040, 0.0200, 0.0240],
      [0.180, 0.0550, 0.0590],
    ],
    innerStructure: {
      type: 'dual_pipe',
      // 下側: 円筒型整流筒 (Ø15 / Ø23)
      bottomCylinder: {
        rIn   : 0.0075, // Ø15mm / 2
        rOut  : 0.0115, // Ø23mm / 2
        zStart: 0.000,
        zEnd  : 0.040,
      },
      // 上側: 中空コーン型整流筒 (z=0.04でØ15/Ø23 → z=0.18でØ85/Ø93)
      topCone: {
        rInStart : 0.0075,
        rOutStart: 0.0115,
        rInEnd   : 0.0425, // Ø85mm / 2
        rOutEnd  : 0.0465, // Ø93mm / 2
        zStart   : 0.040,
        zEnd     : 0.180,
      },
    },
    outlet: {
      innerR  : 0.0425,
      type    : 'dual_pipe',      // 中心流路(Ø15) + 環状流路(Ø23〜Ø85)
      hasCore : true,
      coreR   : 0.0115,           // 下部円筒整流筒の外径半径
      coreInR : 0.0075,           // 下部円筒整流筒の内径半径(中空)
    },
  },

  3: {
    id   : 3,
    name : '図3',
    label: '広口テーパー (上部整流円筒付)',
    desc : '下部テーパーが広い拡散型 (Ø95/Ø103) + 上部整流円筒(Ø40/Ø48)',
    color: '#FFB74D',
    colorHex: 0xFFB74D,
    profile: [
      [0.000, 0.0475, 0.0515],   // 下端: Ø95 / Ø103
      [0.040, 0.0200, 0.0240],
      [0.180, 0.0550, 0.0590],
    ],
    innerStructure: {
      type: 'rectifying_cylinder',
      rIn : 0.0200,
      rOut: 0.0240,
      zStart: 0.040,
      zEnd  : 0.075,
    },
    outlet: {
      innerR  : 0.0475,
      type    : 'uniform',
      hasCore : false,
      coreR   : 0,
    },
  },

  4: {
    id   : 4,
    name : '図4',
    label: '細口テーパー (上部整流円筒付)',
    desc : '下部テーパーが細い集中型 (Ø70/Ø78) + 上部整流円筒(Ø40/Ø48)',
    color: '#F48FB1',
    colorHex: 0xF48FB1,
    profile: [
      [0.000, 0.0350, 0.0390],   // 下端: Ø70 / Ø78
      [0.040, 0.0200, 0.0240],
      [0.180, 0.0550, 0.0590],
    ],
    innerStructure: {
      type: 'rectifying_cylinder',
      rIn : 0.0200,
      rOut: 0.0240,
      zStart: 0.040,
      zEnd  : 0.075,
    },
    outlet: {
      innerR  : 0.0350,
      type    : 'uniform',
      hasCore : false,
      coreR   : 0,
    },
  },
};

// 出口速度の設定関数
// 9.6m/sはノズル上部(Ø110mm)の入口条件。ノズル下端は質量流量を
// 保存するため、形状ごとの出口流路面積から速度を算出する。
function updateNozzleVelocities(inletVelocity, inletTemp) {
  if (typeof inletVelocity === 'number' && !isNaN(inletVelocity) && inletVelocity > 0) {
    AC.inletVelocity = inletVelocity;
  }
  if (typeof inletTemp === 'number' && !isNaN(inletTemp)) {
    AC.inletTemp = inletTemp;
  }
  AC.massFlow = AC.rho * AC.inletVelocity * AC.inletArea;
  for (const n of Object.values(NOZZLES)) {
    const r = n.outlet.innerR;
    const rc = n.outlet.coreR;
    let A = Math.PI * r * r;
    if (n.outlet.hasCore) A -= Math.PI * rc * rc;
    n.outlet.area     = A;
    n.outlet.velocity = AC.inletVelocity * AC.inletArea / A;
  }
}
updateNozzleVelocities();

// ─────────────────────────────────────────────────────────────────
//  ユーティリティ関数
// ─────────────────────────────────────────────────────────────────

/**
 * ノズル断面の内外径を高さから線形補間で取得
 * @param {number} nozzleId - 1〜4
 * @param {number} z        - 底端からの高さ [m] (0〜0.18)
 * @returns {{ rInner, rOuter }}
 */
function getNozzleSection(nozzleId, z) {
  const profile = NOZZLES[nozzleId].profile;
  z = Math.max(0, Math.min(0.18, z));
  for (let i = 0; i < profile.length - 1; i++) {
    const [z0, ri0, ro0] = profile[i];
    const [z1, ri1, ro1] = profile[i + 1];
    if (z >= z0 && z <= z1) {
      const t = (z - z0) / (z1 - z0);
      return { rInner: ri0 + t * (ri1 - ri0), rOuter: ro0 + t * (ro1 - ro0) };
    }
  }
  const last = profile[profile.length - 1];
  return { rInner: last[1], rOuter: last[2] };
}

/**
 * ノズル内部の整流構造（内管・二重管・整流円筒）の寸法を取得
 * @param {number} nozzleId - 1〜4
 * @param {number} zPhys    - ノズル底端からの高さ [m] (0〜0.18)
 * @returns {{ hasInner: boolean, rIn: number, rOut: number }}
 */
function getNozzleInnerStructure(nozzleId, zPhys) {
  const z = Math.max(0, Math.min(0.18, zPhys));
  if (nozzleId === 2) {
    // 図2: 二重管構造（下部円筒型整流筒 + 上部中空コーン型整流筒）
    if (z <= 0.040) {
      // 下側円筒型整流筒: 内径 Ø15mm (r=0.0075) / 外径 Ø23mm (r=0.0115)
      return { hasInner: true, rIn: 0.0075, rOut: 0.0115 };
    } else {
      // 上部中空コーン型整流筒: z=0.04でØ15/Ø23 → z=0.18でØ85/Ø93
      const t = (z - 0.040) / (0.180 - 0.040);
      const rIn  = 0.0075 + (0.0425 - 0.0075) * t; // Ø15mm → Ø85mm
      const rOut = 0.0115 + (0.0465 - 0.0115) * t; // Ø23mm → Ø93mm
      return { hasInner: true, rIn, rOut };
    }
  } else {
    // 図1, 図3, 図4: 喉部(z=0.040)から上部側に伸びる円筒形整流筒 (内径 Ø40mm / 外径 Ø48mm)
    // 高さ: z = 0.040 〜 0.075m (約35mm長)
    if (z >= 0.040 && z <= 0.075) {
      return { hasInner: true, rIn: 0.0200, rOut: 0.0240 };
    }
  }
  return { hasInner: false, rIn: 0, rOut: 0 };
}

/**
 * 任意の位置 (zPhys, r) がノズルの壁面（外管肉部または内部シリンダー・整流筒肉部）か判定
 * @param {number} nozzleId
 * @param {number} zPhys - 底端からの高さ [m] (0〜0.18)
 * @param {number} r     - 中心軸からの半径 [m]
 * @returns {boolean} true: 壁 (WALL), false: 流体 (FLUID) またはノズル外部
 */
function isNozzleWall(nozzleId, zPhys, r) {
  const sec = getNozzleSection(nozzleId, zPhys);
  if (r > sec.rOuter) return false;  // ノズル外側
  if (r >= sec.rInner) return true;  // 外管の肉部
  const inner = getNozzleInnerStructure(nozzleId, zPhys);
  if (inner.hasInner && r >= inner.rIn && r <= inner.rOut) {
    return true; // 内部シリンダー・整流筒の肉部
  }
  return false; // 流体（中空内部、または外側環状部）
}

/**
 * ノズルのワイヤーフレームライン群（外管＋内部二重管・整流筒）を生成
 * @param {number} nozzleId
 * @param {number} zBase - 底端の z 座標（例: 1.0 または 0.0）
 * @returns {number[]} [x0,y0,z0, x1,y1,z1, ...]
 */
function getNozzleWireframeLines(nozzleId, zBase = 1.0) {
  const lines = [];
  const add = (a, b) => lines.push(...a, ...b);
  const segments = 24;

  // 1. 外管の円周リング
  const levels = 12;
  for (let iz = 0; iz <= levels; iz++) {
    const z = 0.18 * iz / levels;
    const s = getNozzleSection(nozzleId, z);
    for (let q = 0; q < segments; q++) {
      const a = 2 * Math.PI * q / segments, b = 2 * Math.PI * (q + 1) / segments;
      add([s.rOuter * Math.cos(a), z + zBase, s.rOuter * Math.sin(a)],
          [s.rOuter * Math.cos(b), z + zBase, s.rOuter * Math.sin(b)]);
      if (iz === 0 || iz === Math.round(levels * 0.04 / 0.18) || iz === levels) {
        add([s.rInner * Math.cos(a), z + zBase, s.rInner * Math.sin(a)],
            [s.rInner * Math.cos(b), z + zBase, s.rInner * Math.sin(b)]);
      }
    }
  }
  // 外管の縦母線 (4方向)
  for (let q = 0; q < segments; q += segments / 4) {
    const a = 2 * Math.PI * q / segments;
    for (let iz = 1; iz <= levels; iz++) {
      const zPrev = 0.18 * (iz - 1) / levels, z = 0.18 * iz / levels;
      const rPrev = getNozzleSection(nozzleId, zPrev).rOuter;
      const r = getNozzleSection(nozzleId, z).rOuter;
      add([rPrev * Math.cos(a), zPrev + zBase, rPrev * Math.sin(a)],
          [r * Math.cos(a), z + zBase, r * Math.sin(a)]);
    }
  }

  // 2. 内部整流構造（内管・二重管・整流円筒）のワイヤーフレーム
  if (nozzleId === 2) {
    // 図2: 下部円筒 (z: 0〜0.04) + 上部中空コーン (z: 0.04〜0.18)
    const innerLevels = 10;
    for (let iz = 0; iz <= innerLevels; iz++) {
      const z = 0.18 * iz / innerLevels;
      const inner = getNozzleInnerStructure(nozzleId, z);
      if (inner.hasInner) {
        for (let q = 0; q < segments; q++) {
          const a = 2 * Math.PI * q / segments, b = 2 * Math.PI * (q + 1) / segments;
          add([inner.rOut * Math.cos(a), z + zBase, inner.rOut * Math.sin(a)],
              [inner.rOut * Math.cos(b), z + zBase, inner.rOut * Math.sin(b)]);
          if (iz === 0 || iz === Math.round(innerLevels * 0.04 / 0.18) || iz === innerLevels) {
            add([inner.rIn * Math.cos(a), z + zBase, inner.rIn * Math.sin(a)],
                [inner.rIn * Math.cos(b), z + zBase, inner.rIn * Math.sin(b)]);
          }
        }
      }
    }
    // 内部シリンダーの縦母線 (4方向)
    for (let q = 0; q < segments; q += segments / 4) {
      const a = 2 * Math.PI * q / segments;
      for (let iz = 1; iz <= innerLevels; iz++) {
        const zPrev = 0.18 * (iz - 1) / innerLevels, z = 0.18 * iz / innerLevels;
        const inPrev = getNozzleInnerStructure(nozzleId, zPrev);
        const inCurr = getNozzleInnerStructure(nozzleId, z);
        if (inPrev.hasInner && inCurr.hasInner) {
          add([inPrev.rOut * Math.cos(a), zPrev + zBase, inPrev.rOut * Math.sin(a)],
              [inCurr.rOut * Math.cos(a), z + zBase, inCurr.rOut * Math.sin(a)]);
        }
      }
    }
  } else {
    // 図1, 3, 4: 上部整流円筒 (z: 0.040 〜 0.075)
    const z0 = 0.040, z1 = 0.075;
    const inner = getNozzleInnerStructure(nozzleId, z0);
    if (inner.hasInner) {
      for (const z of [z0, (z0 + z1) * 0.5, z1]) {
        for (let q = 0; q < segments; q++) {
          const a = 2 * Math.PI * q / segments, b = 2 * Math.PI * (q + 1) / segments;
          add([inner.rOut * Math.cos(a), z + zBase, inner.rOut * Math.sin(a)],
              [inner.rOut * Math.cos(b), z + zBase, inner.rOut * Math.sin(b)]);
          if (z === z0 || z === z1) {
            add([inner.rIn * Math.cos(a), z + zBase, inner.rIn * Math.sin(a)],
                [inner.rIn * Math.cos(b), z + zBase, inner.rIn * Math.sin(b)]);
          }
        }
      }
      for (let q = 0; q < segments; q += segments / 4) {
        const a = 2 * Math.PI * q / segments;
        add([inner.rOut * Math.cos(a), z0 + zBase, inner.rOut * Math.sin(a)],
            [inner.rOut * Math.cos(a), z1 + zBase, inner.rOut * Math.sin(a)]);
      }
    }
  }

  return lines;
}

// ─────────────────────────────────────────────────────────────────
//  非一様格子(ストレッチ格子)ユーティリティ
//  tanh(双曲線正接)クラスタリングにより、指定領域の一部だけを
//  高解像度化する。トポロジー(添字ベース近傍・i,j,k構造)はそのまま。
// ─────────────────────────────────────────────────────────────────

/**
 * 1軸分のストレッチ格子(セル中心座標・セル境界座標)を生成する。
 * @param {number} NX   - セル数
 * @param {number} L    - 軸方向の物理長さ [m] (0〜L)
 * @param {number} beta - ストレッチ強度。大きいほど強く偏り、0に近いほど一様。
 *                        目安: 1.0(緩い)〜3.0(強い)。0や負値は不可。
 * @param {'center'|'edge-high'|'edge-low'|'uniform'} mode
 *        'center'    : 軸の中央(L/2)を高解像度化 (例: ノズル軸 x=y=1.0m)
 *        'edge-high' : 高index側の端(t=1)を高解像度化 (例: 天井付近)
 *        'edge-low'  : 低index側の端(t=0)を高解像度化 (例: 床付近)
 *        'uniform'   : 従来通りの一様格子(デバッグ・比較用)
 * @returns {{ centers: Float64Array, faces: Float64Array, h: Float64Array }}
 *          centers: セル中心座標 (長さNX)
 *          faces  : セル境界座標 (長さNX+1、0〜L)
 *          h      : 各セルの幅 = faces[i+1]-faces[i] (長さNX、参考値)
 */
function buildStretchedAxis(NX, L, beta = 2.0, mode = 'center') {
  const b = Math.max(1e-3, beta); // tanh(0)=0 での0除算防止
  const invTanhB = 1 / Math.tanh(b);
  const faces = new Float64Array(NX + 1);
  for (let i = 0; i <= NX; i++) {
    const t = i / NX; // 一様パラメータ 0〜1
    let s;
    switch (mode) {
      case 'center':
        // t=0.5 (軸中央) 付近を密に、両端を粗くする対称ストレッチ
        s = 0.5 * (Math.tanh(b * (2 * t - 1)) * invTanhB + 1);
        break;
      case 'edge-high':
        // t=1 (高index端) を密にする片側ストレッチ
        s = 1 - Math.tanh(b * (1 - t)) * invTanhB;
        break;
      case 'edge-low':
        // t=0 (低index端) を密にする片側ストレッチ
        s = Math.tanh(b * t) * invTanhB;
        break;
      default:
        s = t; // 'uniform'
    }
    faces[i] = s * L;
  }
  const centers = new Float64Array(NX);
  const h = new Float64Array(NX);
  for (let i = 0; i < NX; i++) {
    centers[i] = 0.5 * (faces[i] + faces[i + 1]);
    h[i] = faces[i + 1] - faces[i];
  }
  return { centers, faces, h };
}

/**
 * FVMソルバー用に x, y, z 3軸分のストレッチ格子と、非一様ステンシルで
 * 使う前後セル間距離(hE/hW/hN/hS/hU/hD)をまとめて生成する。
 * x, y は室内が正方形(0〜DOM)でノズル軸が中心(DOM/2, DOM/2)にあるため
 * 同一のストレッチを共有する。z は天井(k=NX-1)側にノズル・吹出し流が
 * あるため、高index側を密にする。
 * @param {number} NX
 * @param {number} DOM      - x,y方向の室内一辺 [m]
 * @param {object} [opts]
 * @param {number} [opts.betaXY=2.0]
 * @param {number} [opts.betaZ=1.6]
 * @param {number} [opts.domZ=DOM] - z方向の物理長さ(通常DOMと同じ)
 */
function buildFVMGridAxes(NX, DOM, opts = {}) {
  const betaXY = opts.betaXY ?? 2.0;
  const betaZ  = opts.betaZ  ?? 1.6;
  const domZ   = opts.domZ  ?? DOM;

  const axX = buildStretchedAxis(NX, DOM, betaXY, 'center');
  const axY = axX; // x,yは対称なので共有(必要なら個別生成も可)
  const axZ = buildStretchedAxis(NX, domZ, betaZ, 'edge-high');

  // 非一様ステンシル用: セルiの東西(西=i-1側)/南北/上下 距離
  // 境界(i=0のhW, i=NX-1のhE)は実際には内部ループ(1..NX-2)では
  // 参照されないが、安全のため隣接値で埋めておく。
  const mkEW = (ax) => {
    const hE = new Float64Array(NX), hW = new Float64Array(NX);
    for (let i = 0; i < NX - 1; i++) hE[i] = ax.centers[i + 1] - ax.centers[i];
    for (let i = 1; i < NX; i++) hW[i] = ax.centers[i] - ax.centers[i - 1];
    hE[NX - 1] = hE[NX - 2]; hW[0] = hW[1];
    return { hE, hW };
  };

  const { hE: hxE, hW: hxW } = mkEW(axX);
  const { hE: hyN, hW: hyS } = mkEW(axY);
  const { hE: hzU, hW: hzD } = mkEW(axZ);

  return {
    x: axX.centers, y: axY.centers, z: axZ.centers,
    faceX: axX.faces, faceY: axY.faces, faceZ: axZ.faces,
    // 各セルの物理的な幅(可視化のボックスサイズ等に使用。
    // hxE/hxW(隣接セル中心間距離)とは近いが厳密には別の量)
    wx: axX.h, wy: axY.h, wz: axZ.h,
    hxE, hxW, hyN, hyS, hzU, hzD,
  };
}

/**
 * SVG ノズル断面図を生成（コントロールパネル用）
 * @param {number} nozzleId
 * @param {number} width  px
 * @param {number} height px
 * @returns {string} SVG文字列
 */
function nozzleSVG(nozzleId, width = 80, height = 100) {
  const nozzle = NOZZLES[nozzleId];
  const nSteps = 24;
  const cx = width / 2;
  const scale = (width * 0.45) / 0.059;  // Ø118mm を幅の90%に
  const yTop = 4, yBot = height - 4;
  const H = yBot - yTop;

  // 外壁右側の点列 (底→上)
  const outer = [], inner = [];
  for (let i = 0; i <= nSteps; i++) {
    const t = i / nSteps;
    const z = 0.18 * t;
    const { rInner, rOuter } = getNozzleSection(nozzleId, z);
    const y = yBot - H * t;
    outer.push(`${(cx + rOuter * scale).toFixed(1)},${y.toFixed(1)}`);
    inner.push(`${(cx + rInner * scale).toFixed(1)},${y.toFixed(1)}`);
  }
  // 左側（鏡像）
  const outerL = outer.map(pt => {
    const [x, y] = pt.split(',');
    return `${(2 * cx - parseFloat(x)).toFixed(1)},${y}`;
  });

  const outerPoly = [...outer, ...outerL.reverse()].join(' ');
  const innerPoly = [...inner.slice().reverse(), ...inner.map(pt => {
    const [x, y] = pt.split(',');
    return `${(2 * cx - parseFloat(x)).toFixed(1)},${y}`;
  })].join(' ');

  return `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
    <polygon points="${outerPoly}" fill="${nozzle.color}" fill-opacity="0.3" stroke="${nozzle.color}" stroke-width="1.2"/>
    <polygon points="${innerPoly}" fill="#1a1a2e" fill-opacity="0.8" stroke="${nozzle.color}" stroke-width="0.8" stroke-dasharray="3,2"/>
    <line x1="${cx}" y1="${yTop}" x2="${cx}" y2="${yBot}" stroke="rgba(255,255,255,0.2)" stroke-width="0.5" stroke-dasharray="2,2"/>
  </svg>`;
}

/**
 * Three.js用ノズルメッシュ生成 (LatheGeometry)
 * @param {number} nozzleId  1〜4
 * @returns {THREE.Group | null}
 */
function createNozzleMesh(nozzleId) {
  if (typeof THREE === 'undefined') return null;
  const nozzle = NOZZLES[nozzleId];
  const colorHex = nozzle.colorHex || 0x4FC3F7;
  const group = new THREE.Group();

  // getNozzleWireframeLines で外管および内部二重管・整流筒の全輪郭線を統一生成
  const linePositions = getNozzleWireframeLines(nozzleId, 0.0);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  const mat = new THREE.LineBasicMaterial({
    color: colorHex,
    transparent: true,
    opacity: 0.65,
    depthWrite: false,
  });
  group.add(new THREE.LineSegments(geo, mat));
  return group;
}

// ─────────────────────────────────────────────────────────────────
//  非均一格子伸縮ユーティリティ (tanh stretching)
//
//  目的: ノズル軸(center付近)に格子点を集中させ、外壁付近を粗くすることで、
//        均一格子よりも少ない格子点数で噴流コアの境界層・せん断層を解像する。
//
//  定式化 (Roberts 1971 型 両端/片端 tanh stretching):
//    区間 [0, L] 上で「中心」center に向かって集中する N 点を返す。
//    内部では区間を [0, center] と [center, L] の2ブロックに分け、
//    各ブロック内で tanh stretching を適用して結合する。
//
//  引数:
//    N      : 格子点数 (エッジを含む)
//    L      : 区間長 [m]
//    center : 集中させる物理座標 [m]  (通常 L/2 = ノズル軸)
//    beta   : 集中パラメータ (推奨: 2.0〜5.0。大きいほど中央に密)
//
//  戻り値: Float64Array[N], 値域 [0, L]
// ─────────────────────────────────────────────────────────────────
function buildStretchedGrid(N, L, center, beta) {
  if (N <= 1) return new Float64Array([center]);
  const g = new Float64Array(N);

  const frac = center / L;
  const nL   = Math.max(1, Math.round(frac * (N - 1)));  // 左側区間数
  const nR   = N - 1 - nL;                              // 右側区間数
  const tanhB = Math.tanh(beta);

  // 左ブロック [0, center]: t=0(左壁)→1(center) で右端(center)に向かって密に
  for (let i = 0; i <= nL; i++) {
    const t = i / nL;
    const s = Math.tanh(beta * t) / tanhB;
    g[i] = s * center;
  }

  // 右ブロック [center, L]: t=0(center)→1(右壁) で左端(center)が最も密、外側に向かって粗く
  for (let i = 1; i <= nR; i++) {
    const t = i / nR;
    const s = 1.0 + Math.tanh(beta * (t - 1.0)) / tanhB;
    g[nL + i] = center + s * (L - center);
  }

  // 端点を厳密に固定 (丸め誤差回避)
  g[0]     = 0.0;
  g[N - 1] = L;
  return g;
}

/**
 * z方向 tanh 格子: 天井(z=L)付近を密に、床付近をやや密に、中央を粗く。
 * ノズル出口(天井)境界層と床面壁の解像を同時に確保する。
 *
 * @param {number} N     格子点数
 * @param {number} L     区間長 [m]
 * @param {number} betaT 天井側集中パラメータ
 * @param {number} betaF 床側集中パラメータ
 * @param {number} [alpha=0.7] 天井側(sT)と床側(sF)のブレンド比率 (0〜1、大きいほど天井側優先)。
 *   天井直下セル厚は betaT を上げるだけでは頭打ちになりやすく、alpha を1に近づける方が効く。
 *   既定値0.7は既存呼び出し元との後方互換のため維持。
 * @returns {Float64Array}
 */
function buildStretchedZGrid(N, L, betaT, betaF, alpha = 0.7) {
  if (N <= 1) return new Float64Array([0]);
  const g     = new Float64Array(N);
  const tanhT = Math.tanh(betaT);
  const tanhF = Math.tanh(betaF);

  for (let k = 0; k < N; k++) {
    const t  = k / (N - 1);                                   // 0(床) → 1(天井)
    const sT = Math.tanh(betaT * t) / tanhT;                  // 天井集中 (t=1で密)
    const sF = 1.0 + Math.tanh(betaF * (t - 1.0)) / tanhF;  // 床集中 (t=0で密)
    g[k] = (alpha * sT + (1.0 - alpha) * sF) * L;
  }
  g[0]     = 0.0;
  g[N - 1] = L;
  return g;
}

/**
 * 単調増加配列 arr 中で val 以下の最大インデックスを二分探索で返す。
 * 非均一格子で物理座標 → セル添字の逆引きに使用。
 * @param {Float64Array|Float32Array} arr
 * @param {number} val
 * @returns {number} 0 ～ arr.length-2 (クランプ済み)
 */
function bisectLeft(arr, val) {
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (arr[mid] <= val) lo = mid;
    else hi = mid - 1;
  }
  return Math.max(0, Math.min(arr.length - 2, lo));
}
