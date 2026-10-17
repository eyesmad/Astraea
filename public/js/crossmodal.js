// Part 3: cross-modal consistency. Mouth aperture (MediaPipe) vs audio energy envelope (Web Audio).
// Both tracks share the performance.now() clock, so the lag at peak correlation is the audio/video sync offset.
import { crossCorrelate, movingAverage, resample, zscore } from './crosscorr.js';
import { clamp, median, std, trimWindow } from './stats.js';

// Camera and microphone pipelines have their own latencies, so a real call almost always shows a constant
// audio/video offset of 100 ms or more. That is a property of the devices, not evidence of manipulation:
// judging the raw offset flagged every honest webcam. What does indicate manipulation is the offset moving
// (drift) or the tracks not corresponding at all (decoupling), so the baseline is learned first and the
// signal measures deviation from it. A gross offset is still reported outright, because no capture pipeline
// explains most of a second of skew.
export const BASELINE_MIN_SAMPLES = 6;
export const BASELINE_MIN_PEAK = 0.2;
export const GROSS_OFFSET_MS = 600;
export const DECOUPLED_RUN = 3; // consecutive low-correlation measurements before LIP_DECOUPLED fires

export class CrossModal {
  constructor({ windowMs = 8000, hz = 25, maxLagMs = 800, driftMs = 150 } = {}) {
    this.windowMs = windowMs;
    this.hz = hz;
    this.maxLagMs = maxLagMs;
    this.driftMs = driftMs;
    this.reset();
  }

  reset() {
    this.mouth = [];
    this.audio = [];
    this.lagHistory = [];
    this.baselineSamples = [];
    this.baseline = null;
    this.lowCorrRun = 0;
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

  // Live read-out for the overlay. See VideoMetrics.probe for the shape.
  probe(now) {
    const r = this.measure(now);
    const hasBoth = Boolean(this.mouth.length && this.audio.length);
    let text = 'needs a speaking face and audible voice';
    let bar = 0;
    if (r && this.baseline != null) {
      const dev = r.lagMs - this.baseline;
      const base = this.baseline >= 0 ? `+${this.baseline.toFixed(0)}` : this.baseline.toFixed(0);
      text = `${dev >= 0 ? '+' : ''}${dev.toFixed(0)} ms from this device's ${base} ms baseline`;
      bar = clamp(Math.abs(dev) / 400, 0, 1);
    } else if (r) {
      text = `calibrating, ${this.baselineSamples.length}/${BASELINE_MIN_SAMPLES} samples (raw ${r.lagMs >= 0 ? '+' : ''}${r.lagMs.toFixed(0)} ms)`;
    }
    return [
      {
        key: 'sync', label: 'Lip-sync offset',
        progress: r ? (this.baseline != null ? 1 : 0.5) : (hasBoth ? 0.6 : 0),
        text,
        bar,
      },
      {
        key: 'corr', label: 'Lip/voice correlation', progress: r ? 1 : 0,
        text: r ? r.peak.toFixed(2) : 'measuring',
        bar: r ? clamp(1 - r.peak / 0.5, 0, 1) : 0,
      },
    ];
  }

  _updateBaseline(r) {
    if (this.baseline != null || r.peak < BASELINE_MIN_PEAK) return;
    this.baselineSamples.push(r.lagMs);
    if (this.baselineSamples.length >= BASELINE_MIN_SAMPLES) this.baseline = median(this.baselineSamples);
  }

  evaluate(now) {
    const r = this.measure(now);
    if (!r) return [];
    this.last = { ...r, t: now };
    this._updateBaseline(r);
    this.lagHistory.push(r.lagMs);
    if (this.lagHistory.length > 8) this.lagHistory.shift();

    this.lowCorrRun = r.peak < 0.12 ? this.lowCorrRun + 1 : 0;
    const decoupled = this.lowCorrRun >= DECOUPLED_RUN;

    const range = this.lagHistory.length >= 5 ? Math.max(...this.lagHistory) - Math.min(...this.lagHistory) : 0;
    const drifting = range >= this.driftMs && r.peak >= 0.2;
    const gross = Math.abs(r.lagMs) >= GROSS_OFFSET_MS && r.peak >= 0.25;

    const out = [];
    if (this.baseline == null && !gross) {
      out.push({
        id: 'LIP_DRIFT',
        fired: false,
        severity: 0,
        detail: `Learning this device's audio/video offset (${this.baselineSamples.length}/${BASELINE_MIN_SAMPLES} samples).`,
      });
    } else {
      const dev = r.lagMs - (this.baseline ?? 0);
      const shifted = Math.abs(dev) >= this.driftMs && r.peak >= 0.25;
      const fired = shifted || drifting || gross;
      const base = (this.baseline ?? 0) >= 0 ? `+${(this.baseline ?? 0).toFixed(0)}` : (this.baseline ?? 0).toFixed(0);
      let why;
      if (gross) why = `Audio ${r.lagMs >= 0 ? 'lags' : 'leads'} the mouth by ${Math.abs(r.lagMs).toFixed(0)} ms, far beyond any capture delay`;
      else if (shifted) why = `Sync shifted ${Math.abs(dev).toFixed(0)} ms from this device's ${base} ms baseline`;
      else if (drifting) why = `Sync wandered ${range.toFixed(0)} ms`;
      else why = `Audio ${r.lagMs >= 0 ? 'lags' : 'leads'} the mouth by ${Math.abs(r.lagMs).toFixed(0)} ms, in line with this device's ${base} ms baseline`;
      out.push({
        id: 'LIP_DRIFT',
        fired,
        severity: fired ? clamp(Math.max(Math.abs(dev), range, gross ? Math.abs(r.lagMs) : 0) / 400, 0.4, 1) : 0,
        detail: `${why} (peak correlation ${r.peak.toFixed(2)}).`,
      });
    }
    out.push({
      id: 'LIP_DECOUPLED',
      fired: decoupled,
      severity: decoupled ? 0.5 : 0,
      detail: decoupled
        ? `Lip movement and voice have not corresponded for ${this.lowCorrRun} readings (${r.peak.toFixed(2)}).`
        : `Lips and voice correlate (${r.peak.toFixed(2)}).`,
    });
    return out;
  }
}
