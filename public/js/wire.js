// Wire format shared by the browser and the server for streaming face data (MediaPipe stays in the browser;
// only the 19 landmarks and 12 blendshapes the metrics read are sent).
import { BLEND_KEYS, IDX } from './videoMetrics.js';

export const LM_IDS = [...new Set([...IDX.rightEye, ...IDX.leftEye, IDX.noseTip, IDX.forehead, IDX.chin, IDX.lipTop, IDX.lipBottom, ...IDX.stable])];

const r5 = (v) => Math.round(v * 1e5) / 1e5;

export function packFace(lm, blend) {
  const p = [];
  for (const id of LM_IDS) p.push(r5(lm[id].x), r5(lm[id].y));
  if (blend) for (const k of BLEND_KEYS) p.push(r5(blend[k] ?? 0));
  return p;
}

export function unpackFace(p) {
  if (!Array.isArray(p) || p.length < LM_IDS.length * 2 || p.some((v) => typeof v !== 'number' || !Number.isFinite(v))) return null;
  const lm = [];
  LM_IDS.forEach((id, i) => { lm[id] = { x: p[2 * i], y: p[2 * i + 1] }; });
  let blend = null;
  if (p.length >= LM_IDS.length * 2 + BLEND_KEYS.length) {
    blend = {};
    BLEND_KEYS.forEach((k, i) => { blend[k] = p[LM_IDS.length * 2 + i]; });
  }
  return { lm, blend };
}
