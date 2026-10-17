// Part 3: cross-modal consistency. Mouth aperture (MediaPipe) vs audio energy envelope (Web Audio).
// Both tracks share the performance.now() clock, so the lag at peak correlation is the audio/video sync offset.
import { crossCorrelate, movingAverage, resample, zscore } from './crosscorr.js';
import { clamp, std, trimWindow } from './stats.js';

export class CrossModal {
  constructor({ windowMs = 8000, hz = 25, maxLagMs = 500, driftMs = 150 } = {}) {
    this.windowMs = windowMs;
    this.hz = hz;
    this.maxLagMs = maxLagMs;
    this.driftMs = driftMs;
    this.mouth = [];
    this.audio = [];
    this.lagHistory = [];
    this.last = null;
  }

  reset() {
    this.mouth = [];
    this.audio = [];
    this.lagHistory = [];
    this.last = null;
  }

  pushMouth(t, v) {
    this.mouth.push({ t, v });
    trimWindow(this.mouth, t, this.windowMs + 2000);
  }

  pushAudio(t, v) {
    this.audio.push({ t, v });
    trimWindow(this.audio, t, this.windowMs + 2000);
  }

  // Returns { lagMs, peak } or null if there is not enough activity to judge.
  measure(now) {
    if (!this.mouth.length || !this.audio.length) return null;
    // End the window at the newest sample both tracks have; sampling up to `now` would run past the data.
    const t1 = Math.min(this.mouth[this.mouth.length - 1].t, this.audio[this.audio.length - 1].t);
    if (now - t1 > 1500) return null; // stale: capture stopped or no face
    const t0 = t1 - this.windowMs;
    const m = resample(this.mouth, this.hz, t0, t1);
    const a = resample(this.audio, this.hz, t0, t1);
    if (m.length < this.hz * 5 || m.some((v) => v == null) || a.some((v) => v == null)) return null;
    const ms = movingAverage(m, 2);
    const as = movingAverage(a, 3);
    // Need both tracks genuinely active: a speaking mouth and speech-level audio.
    if (std(ms) < 0.004 || std(as) < 0.004) return null;
    const maxLag = Math.round((this.maxLagMs / 1000) * this.hz);
    const { lag, corr } = crossCorrelate(zscore(ms), zscore(as), maxLag);
    return { lagMs: (lag / this.hz) * 1000, peak: corr };
  }

  evaluate(now) {
    const r = this.measure(now);
    if (!r) return [];
    this.last = { ...r, t: now };
    this.lagHistory.push(r.lagMs);
    if (this.lagHistory.length > 8) this.lagHistory.shift();

    const out = [];
    const range = this.lagHistory.length >= 5 ? Math.max(...this.lagHistory) - Math.min(...this.lagHistory) : 0;
    const offset = Math.abs(r.lagMs) >= this.driftMs && r.peak >= 0.25;
    const drifting = range >= this.driftMs && r.peak >= 0.2;

    out.push({
      id: 'LIP_DRIFT',
      fired: offset || drifting,
      severity: offset || drifting ? clamp(Math.max(Math.abs(r.lagMs), range) / 400, 0.4, 1) : 0,
      detail: `Audio ${r.lagMs >= 0 ? 'lags' : 'leads'} the mouth by ${Math.abs(r.lagMs).toFixed(0)} ms (peak correlation ${r.peak.toFixed(2)}${drifting ? `, lag wandered ${range.toFixed(0)} ms` : ''}).`,
    });
    out.push({
      id: 'LIP_DECOUPLED',
      fired: r.peak < 0.12,
      severity: r.peak < 0.12 ? 0.5 : 0,
      detail: r.peak < 0.12 ? `Lip movement and voice barely correlate (${r.peak.toFixed(2)}).` : `Lips and voice correlate (${r.peak.toFixed(2)}).`,
    });
    return out;
  }
}
