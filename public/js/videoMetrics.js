// Part 2 (video): blink cadence, motion smoothness and expression variance from MediaPipe Face Mesh landmarks.
// Thresholds are heuristics for a proof of concept; calibrate them against real call footage.
import { clamp, dist, mean, median, std, trimWindow } from './stats.js';

export const IDX = {
  rightEye: [33, 160, 158, 133, 153, 144],
  leftEye: [362, 385, 387, 263, 373, 380],
  noseTip: 1,
  forehead: 10,
  chin: 152,
  lipTop: 13,
  lipBottom: 14,
  stable: [1, 152, 33, 263, 61, 291, 10],
};

export const BLEND_KEYS = [
  'jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'browDownLeft', 'browDownRight',
  'browInnerUp', 'eyeSquintLeft', 'eyeSquintRight', 'mouthFrownLeft', 'mouthFrownRight', 'cheekPuff', 'mouthPucker',
];

export function eyeAspect(lm, ids) {
  const [p1, p2, p3, p4, p5, p6] = ids.map((i) => lm[i]);
  return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4) || 1);
}

export function mouthAperture(lm) {
  return dist(lm[IDX.lipTop], lm[IDX.lipBottom]) / (dist(lm[IDX.forehead], lm[IDX.chin]) || 1);
}

export class VideoMetrics {
  constructor() {
    this.reset();
  }

  reset() {
    this.firstT = null;
    this.lastT = null;
    this.earHist = [];
    this.closedAt = null;
    this.blinks = []; // timestamps (ms)
    this.pose = []; // { t, v: number[] } normalised stable landmarks
    this.blend = []; // { t, v: number[] }
    this.mouth = []; // { t, v }
    this.frames = 0;
  }

  // lm: array of {x,y,z} normalised landmarks. blend: { categoryName: score } or null.
  push(t, lm, blend) {
    if (!lm) return;
    this.frames++;
    if (this.firstT == null) this.firstT = t;
    this.lastT = t;

    this.mouth.push({ t, v: mouthAperture(lm) });
    trimWindow(this.mouth, t, 20000);

    const ear = (eyeAspect(lm, IDX.rightEye) + eyeAspect(lm, IDX.leftEye)) / 2;
    this.earHist.push({ t, v: ear });
    trimWindow(this.earHist, t, 15000);
    this._blinkStep(t, ear);

    const iod = dist(lm[33], lm[263]) || 1;
    const c = IDX.stable.reduce((a, i) => ({ x: a.x + lm[i].x / IDX.stable.length, y: a.y + lm[i].y / IDX.stable.length }), { x: 0, y: 0 });
    const v = [];
    for (const i of IDX.stable) v.push((lm[i].x - c.x) / iod, (lm[i].y - c.y) / iod);
    this.pose.push({ t, v });
    trimWindow(this.pose, t, 6000);

    if (blend) {
      this.blend.push({ t, v: BLEND_KEYS.map((k) => blend[k] ?? 0) });
      trimWindow(this.blend, t, 12000);
    }
    this.blinks = this.blinks.filter((b) => b > t - 60000);
  }

  _blinkStep(t, ear) {
    if (this.earHist.length < 20) return;
    const open = median(this.earHist.map((e) => e.v).filter((v) => v > 0)); // median ~ open-eye level
    const thr = open * 0.72;
    if (ear < thr) {
      if (this.closedAt == null) this.closedAt = t;
    } else if (this.closedAt != null) {
      const d = t - this.closedAt;
      if (d >= 40 && d <= 600) this.blinks.push(this.closedAt);
      this.closedAt = null;
    }
  }

  get observedMs() {
    return this.firstT == null ? 0 : this.lastT - this.firstT;
  }

  blinkRate(now = this.lastT) {
    const win = Math.min(this.observedMs, 45000);
    if (win < 1000) return 0;
    const n = this.blinks.filter((b) => b >= now - win).length;
    return (n / win) * 60000;
  }

  blinkIntervalCV() {
    const b = this.blinks;
    if (b.length < 7) return null;
    const iv = [];
    for (let i = 1; i < b.length; i++) iv.push(b[i] - b[i - 1]);
    return std(iv) / (mean(iv) || 1);
  }

  evaluate() {
    const out = [];
    const obs = this.observedMs;

    // BLINK_CADENCE
    if (obs >= 25000) {
      const rate = this.blinkRate();
      const cv = this.blinkIntervalCV();
      let fired = false;
      let severity = 0;
      let detail = `${rate.toFixed(0)} blinks/min${cv != null ? `, interval variation ${cv.toFixed(2)}` : ''}.`;
      if (rate < 5) {
        fired = true;
        severity = clamp(0.5 + (5 - rate) / 10, 0.5, 0.9);
        detail += ' Far fewer blinks than a person normally makes.';
      } else if (rate > 40) {
        fired = true;
        severity = 0.5;
        detail += ' Unusually frequent blinking.';
      } else if (cv != null && cv < 0.18) {
        fired = true;
        severity = 0.6;
        detail += ' Blinks are metronomic; real blinking is irregular.';
      }
      out.push({ id: 'BLINK_CADENCE', fired, severity, detail });
    }

    // MOTION_SMOOTHNESS: second-difference jitter of face geometry relative to head pose, plus frozen-face check.
    if (this.pose.length >= 45) {
      const p = this.pose;
      let jitter = 0;
      let vel = 0;
      let n = 0;
      for (let i = 2; i < p.length; i++) {
        let j = 0;
        let w = 0;
        for (let k = 0; k < p[i].v.length; k++) {
          const d2 = p[i].v[k] - 2 * p[i - 1].v[k] + p[i - 2].v[k];
          j += d2 * d2;
          const d1 = p[i].v[k] - p[i - 1].v[k];
          w += d1 * d1;
        }
        jitter += j;
        vel += w;
        n++;
      }
      const jr = Math.sqrt(jitter / n / p[0].v.length);
      const vr = Math.sqrt(vel / n / p[0].v.length);
      let fired = false;
      let severity = 0;
      let detail = `Landmark jitter ${(jr * 1000).toFixed(1)}, motion ${(vr * 1000).toFixed(1)} (per-mille of eye distance).`;
      if (jr > 0.012) {
        fired = true;
        severity = clamp(0.4 + (jr - 0.012) * 30, 0.4, 0.9);
        detail += ' Facial geometry flickers between frames.';
      } else if (vr < 0.0003 && this.observedMs > 8000) {
        fired = true;
        severity = 0.35;
        detail += ' Face is unnaturally still.';
      }
      out.push({ id: 'MOTION_SMOOTHNESS', fired, severity, detail });
    }

    // EXPRESSION_FLAT
    if (this.blend.length >= 150 && this.blend[this.blend.length - 1].t - this.blend[0].t >= 9000) {
      const k = this.blend[0].v.length;
      const sds = [];
      for (let c = 0; c < k; c++) sds.push(std(this.blend.map((b) => b.v[c])));
      const avg = mean(sds);
      const fired = avg < 0.008;
      out.push({
        id: 'EXPRESSION_FLAT',
        fired,
        severity: fired ? clamp(0.8 - avg * 50, 0.3, 0.7) : 0,
        detail: `Expression variation ${(avg * 100).toFixed(2)}%${fired ? ': the face barely changes expression.' : '.'}`,
      });
    }
    return out;
  }
}
