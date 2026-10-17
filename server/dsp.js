// Minimal DSP for the server: FFT and an AnalyserNode-compatible dB spectrum so the browser-era feature
// functions (spectralFlatness, brickwallDrop) run unchanged on streamed PCM.

export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

const windows = new Map();
function blackman(n) {
  if (!windows.has(n)) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / n) + 0.08 * Math.cos((4 * Math.PI * i) / n);
    windows.set(n, w);
  }
  return windows.get(n);
}

// Returns n/2 magnitude bins in dB (floored at -120), like AnalyserNode.getFloatFrequencyData.
export function spectrumDb(frame) {
  const n = frame.length;
  const w = blackman(n);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = frame[i] * w[i];
  fft(re, im);
  const out = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) out[i] = Math.max(-120, 20 * Math.log10(Math.hypot(re[i], im[i]) / n + 1e-12));
  return out;
}

// 2:1 decimation with a short windowed-sinc low-pass (for feeding 16 kHz models from 32 kHz capture).
const FIR = (() => {
  const taps = 31;
  const h = new Float64Array(taps);
  let sum = 0;
  for (let i = 0; i < taps; i++) {
    const m = i - (taps - 1) / 2;
    const sinc = m === 0 ? 0.5 : Math.sin(Math.PI * 0.5 * m) / (Math.PI * m);
    h[i] = sinc * (0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (taps - 1)));
    sum += h[i];
  }
  return h.map((v) => v / sum);
})();

export function decimate2(x) {
  const out = new Float32Array(Math.floor(x.length / 2));
  const half = (FIR.length - 1) / 2;
  for (let o = 0; o < out.length; o++) {
    const c = o * 2;
    let s = 0;
    for (let k = 0; k < FIR.length; k++) {
      const idx = c + k - half;
      if (idx >= 0 && idx < x.length) s += FIR[k] * x[idx];
    }
    out[o] = s;
  }
  return out;
}
