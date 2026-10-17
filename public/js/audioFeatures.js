// Frame-level audio features. Pure functions so they can be unit tested without a browser.

export function rms(buf) {
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / buf.length);
}

// Spectral flatness over [loBin, hiBin) of a dB spectrum (as from AnalyserNode.getFloatFrequencyData).
// ~1 for noise, ~0 for tonal/harmonic sounds.
export function spectralFlatness(freqDb, loBin, hiBin) {
  let logSum = 0;
  let sum = 0;
  const n = hiBin - loBin;
  if (n <= 0) return 0;
  for (let i = loBin; i < hiBin; i++) {
    const p = 10 ** (freqDb[i] / 10) + 1e-12;
    logSum += Math.log(p);
    sum += p;
  }
  return Math.exp(logSum / n) / (sum / n);
}

// Normalised autocorrelation pitch estimate. Returns { f0, clarity } or f0 = 0 when unvoiced.
export function estimateF0(buf, sampleRate, fmin = 70, fmax = 400) {
  const minLag = Math.floor(sampleRate / fmax);
  const maxLag = Math.min(Math.floor(sampleRate / fmin), buf.length - 1);
  const n = buf.length - maxLag;
  if (n <= 0) return { f0: 0, clarity: 0 };
  let e0 = 0;
  for (let i = 0; i < n; i++) e0 += buf[i] * buf[i];
  if (e0 < 1e-9) return { f0: 0, clarity: 0 };
  let best = 0;
  let bestLag = 0;
  const r = new Float32Array(maxLag + 2);
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0;
    let e1 = 0;
    for (let i = 0; i < n; i++) {
      s += buf[i] * buf[i + lag];
      e1 += buf[i + lag] * buf[i + lag];
    }
    r[lag] = s / Math.sqrt(e0 * e1 + 1e-12);
  }
  // First strong local maximum avoids octave-down errors.
  const thresh = 0.85 * Math.max(...r.slice(minLag, maxLag + 1));
  for (let lag = minLag + 1; lag < maxLag; lag++) {
    if (r[lag] >= thresh && r[lag] >= r[lag - 1] && r[lag] >= r[lag + 1]) {
      best = r[lag];
      bestLag = lag;
      break;
    }
  }
  if (!bestLag) return { f0: 0, clarity: 0 };
  return { f0: sampleRate / bestLag, clarity: best };
}

// Looks for an abrupt spectral cliff at or above 5 kHz followed by an empty band: the signature of a
// band-limited vocoder or codec. (Narrowband phone audio cuts below 5 kHz and is deliberately not matched.)
export function brickwallDrop(freqDb, sampleRate, fftSize) {
  const binHz = sampleRate / fftSize;
  const bandBins = Math.round(500 / binHz);
  const bands = [];
  for (let start = 0; start + bandBins <= freqDb.length; start += bandBins) {
    let s = 0;
    for (let i = start; i < start + bandBins; i++) s += freqDb[i];
    bands.push({ hz: start * binHz, db: s / bandBins });
  }
  let best = { dropDb: 0, atHz: 0 };
  for (let i = 1; i < bands.length - 2; i++) {
    if (bands[i].hz < 5000) continue;
    const drop = bands[i - 1].db - bands[i].db;
    const tail = Math.max(bands[i + 1].db, bands[i + 2].db);
    if (drop > best.dropDb && bands[i - 1].db - tail > drop * 0.9) best = { dropDb: drop, atHz: bands[i].hz };
  }
  return best;
}
