// Part 3 maths: resample two irregular tracks to a common rate and find the lag of peak cross-correlation.
import { mean, std } from './stats.js';

// track: [{t (ms), v}] sorted by t. Returns a uniform array sampled at `hz` from t0 to t1.
export function resample(track, hz, t0, t1) {
  const out = [];
  if (track.length < 2) return out;
  const step = 1000 / hz;
  let j = 0;
  for (let t = t0; t <= t1; t += step) {
    while (j < track.length - 2 && track[j + 1].t < t) j++;
    const a = track[j];
    const b = track[j + 1];
    if (t < a.t || t > b.t) {
      out.push(null);
      continue;
    }
    const f = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t);
    out.push(a.v + (b.v - a.v) * f);
  }
  return out;
}

export function zscore(arr) {
  const m = mean(arr);
  const s = std(arr) || 1;
  return arr.map((v) => (v - m) / s);
}

export function movingAverage(arr, n) {
  if (n <= 1) return arr.slice();
  const out = new Array(arr.length);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i];
    if (i >= n) sum -= arr[i - n];
    out[i] = sum / Math.min(n, i + 1);
  }
  return out;
}

// Convention: a positive lag means y trails x (y[i + lag] resembles x[i]).
// With x = mouth aperture and y = audio envelope, positive lag means audio arrives late.
export function crossCorrelate(x, y, maxLag) {
  const n = Math.min(x.length, y.length);
  const zx = zscore(x.slice(0, n));
  const zy = zscore(y.slice(0, n));
  const curve = [];
  let best = { lag: 0, corr: -Infinity };
  for (let k = -maxLag; k <= maxLag; k++) {
    let s = 0;
    let c = 0;
    for (let i = 0; i < n; i++) {
      const j = i + k;
      if (j < 0 || j >= n) continue;
      s += zx[i] * zy[j];
      c++;
    }
    const r = c > n / 2 ? s / c : 0;
    curve.push({ lag: k, corr: r });
    if (r > best.corr) best = { lag: k, corr: r };
  }
  return { ...best, curve };
}
