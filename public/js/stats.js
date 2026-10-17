export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function mean(a) {
  if (!a.length) return 0;
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}

export function std(a) {
  if (a.length < 2) return 0;
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) ** 2;
  return Math.sqrt(s / (a.length - 1));
}

export function percentile(a, p) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const i = clamp(p, 0, 1) * (s.length - 1);
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}

export const median = (a) => percentile(a, 0.5);

// Robust spread: interquartile range scaled to match std for a normal distribution.
export const robustStd = (a) => (percentile(a, 0.75) - percentile(a, 0.25)) / 1.349;

export function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Keep only samples newer than (now - windowMs). Mutates and returns the array.
export function trimWindow(arr, now, windowMs) {
  let i = 0;
  while (i < arr.length && arr[i].t < now - windowMs) i++;
  if (i) arr.splice(0, i);
  return arr;
}
