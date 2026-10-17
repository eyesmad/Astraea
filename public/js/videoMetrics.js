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

// MOTION_SMOOTHNESS thresholds. For white landmark noise of standard deviation s (in eye-distance units),
// rms(second difference) is s*sqrt(6) and rms(first difference) is s*sqrt(2), so their ratio tends to sqrt(3)
// = 1.73; smooth real head motion sits well below that because velocity dominates acceleration over 33 ms.
// JITTER_FLOOR therefore says "how much flicker", the ratio says "is it flicker rather than real movement",
// and MIN_MOTION requires the face to actually be moving before either is judged.
// These need calibrating against real call footage; the live read-out in the overlay shows the measured values.
export const JITTER_FLOOR = 0.025;
export const NOISE_RATIO = 1.55;
export const MIN_MOTION = 0.004;

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

  // Flicker of facial geometry relative to head pose. Only consecutive, regularly spaced frames are used:
  // MediaPipe returns nothing when the video frame has not advanced, and a second difference taken across a
  // dropped frame is meaningless (it reads as a large spike), which on its own fired this signal constantly.
  jitterStats() {
    const p = this.pose;
    if (p.length < 45) return null;
    const dts = [];
    for (let i = 1; i < p.length; i++) dts.push(p[i].t - p[i - 1].t);
    const md = median(dts);
    if (!(md > 0)) return null;
    const k = p[0].v.length;
    let jitter = 0;
    let vel = 0;
    let n = 0;
    for (let i = 2; i < p.length; i++) {
      const gapA = p[i - 1].t - p[i - 2].t;
      const gapB = p[i].t - p[i - 1].t;
      if (gapA < md * 0.6 || gapA > md * 1.6 || gapB < md * 0.6 || gapB > md * 1.6) continue;
      for (let c = 0; c < k; c++) {
        const d2 = p[i].v[c] - 2 * p[i - 1].v[c] + p[i - 2].v[c];
        jitter += d2 * d2;
        const d1 = p[i].v[c] - p[i - 1].v[c];
        vel += d1 * d1;
      }
      n++;
    }
    if (n < 30) return null;
    const jr = Math.sqrt(jitter / n / k);
    const vr = Math.sqrt(vel / n / k);
    return { jr, vr, ratio: vr > 0 ? jr / vr : 0, n, dropped: dts.filter((d) => d > md * 1.6).length, fps: 1000 / md };
  }

  // Live read-out for the overlay: what each video probe currently measures, and how close it is to having
  // enough data to judge. `progress` 1 means the signal is being evaluated; `bar` is 0 (normal) to 1 (worst).
  probe() {
    const obs = this.observedMs;
    const js = this.jitterStats();
    const rate = this.blinkRate();
    const cv = this.blinkIntervalCV();
    const blendMs = this.blend.length ? this.blend[this.blend.length - 1].t - this.blend[0].t : 0;
    let expr = null;
    if (this.blend.length >= 20) {
      const k = this.blend[0].v.length;
      const sds = [];
      for (let c = 0; c < k; c++) sds.push(std(this.blend.map((b) => b.v[c])));
      expr = mean(sds);
    }
    return [
      {
        key: 'face', label: 'Face tracking', progress: this.frames ? 1 : 0,
        text: this.frames ? `${this.frames} frames${js ? ` at ${js.fps.toFixed(0)} fps, ${js.dropped} gaps` : ''}` : 'waiting for a face',
        bar: 0,
      },
      {
        key: 'blink', label: 'Blink cadence', progress: clamp(obs / 25000, 0, 1),
        text: obs >= 25000 ? `${rate.toFixed(0)} blinks/min${cv != null ? `, variation ${cv.toFixed(2)}` : ''}` : `${this.blinks.length} blinks so far`,
        bar: clamp(1 - rate / 18, 0, 1),
      },
      {
        key: 'jitter', label: 'Landmark jitter', progress: js ? 1 : clamp(this.pose.length / 45, 0, 1),
        text: js ? `${(js.jr * 1000).toFixed(1)} jitter vs ${(js.vr * 1000).toFixed(1)} motion, ratio ${js.ratio.toFixed(2)}` : 'measuring',
        bar: js ? clamp(js.jr / (JITTER_FLOOR * 2), 0, 1) : 0,
      },
      {
        key: 'expr', label: 'Expression variance', progress: Math.min(clamp(this.blend.length / 150, 0, 1), clamp(blendMs / 9000, 0, 1)),
        text: expr == null ? 'measuring' : `${(expr * 100).toFixed(2)}% average movement`,
        bar: expr == null ? 0 : clamp(1 - expr / 0.02, 0, 1),
      },
    ];
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

    // MOTION_SMOOTHNESS
    const js = this.jitterStats();
    if (js && this.observedMs >= 5000) {
      const { jr, vr, ratio } = js;
      let fired = false;
      let severity = 0;
      let detail = `Landmark jitter ${(jr * 1000).toFixed(1)}, motion ${(vr * 1000).toFixed(1)}, noise ratio ${ratio.toFixed(2)} (per-mille of eye distance).`;
      if (jr > JITTER_FLOOR && ratio > NOISE_RATIO && vr > MIN_MOTION) {
        fired = true;
        severity = clamp(0.4 + (jr - JITTER_FLOOR) * 20, 0.4, 0.9);
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
