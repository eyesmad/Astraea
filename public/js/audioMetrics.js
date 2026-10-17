// Part 2 (voice): vocoder signature, prosodic statistics and respiratory coupling.
// Consumes per-frame features from audioAnalyzer.js. Heuristic thresholds for a proof of concept.
import { clamp, robustStd, trimWindow } from './stats.js';

const SPEECH_RMS = 0.02; // above this: speaking
const DIGITAL_SILENCE = 2e-4; // below this: exactly-silent (TTS, hard gates, aggressive noise suppression)

export class AudioMetrics {
  constructor() {
    this.reset();
  }

  reset() {
    this.firstT = null;
    this.lastT = null;
    this.f0 = []; // { t, v: semitones }
    this.cliff = []; // { t, v: 0|1 }
    this.speechMs = []; // { t, v: ms }
    this.onsets = []; // timestamps of speech onsets
    this.breaths = []; // timestamps of breath events
    this.pauseNoise = []; // { t, v: 0|1 } per pause frame: 1 = digitally silent
    this._state = 'silence';
    this._stateSince = 0;
    this._pauseNoiseMs = 0;
    this._pauseFrames = 0;
    this._pauseSilentFrames = 0;
    this._lastBreathCandidate = false;
  }

  // f: { t, dt, rms, flatness, f0, clarity, dropDb }
  push(f) {
    if (this.firstT == null) this.firstT = f.t;
    this.lastT = f.t;
    const speaking = f.rms >= SPEECH_RMS;

    if (speaking) {
      this.speechMs.push({ t: f.t, v: f.dt });
      if (f.f0 > 0 && f.clarity > 0.6) {
        this.f0.push({ t: f.t, v: 12 * Math.log2(f.f0 / 100) });
        this.cliff.push({ t: f.t, v: f.dropDb >= 40 ? 1 : 0 });
      }
    }
    trimWindow(this.speechMs, f.t, 30000);
    trimWindow(this.f0, f.t, 12000);
    trimWindow(this.cliff, f.t, 10000);

    // Phrase / pause state machine with breath detection inside pauses.
    if (speaking && this._state !== 'speech') {
      if (this._state === 'pause' && f.t - this._stateSince > 150) this.onsets.push(f.t);
      this._state = 'speech';
      this._stateSince = f.t;
      this._lastBreathCandidate = false;
    } else if (!speaking) {
      if (this._state !== 'pause') {
        this._state = 'pause';
        this._stateSince = f.t;
        this._pauseNoiseMs = 0;
      }
      const noisy = f.rms >= 0.0015 && f.rms < SPEECH_RMS && f.flatness > 0.25;
      this._pauseFrames++;
      if (f.rms < DIGITAL_SILENCE) this._pauseSilentFrames++;
      this.pauseNoise.push({ t: f.t, v: f.rms < DIGITAL_SILENCE ? 1 : 0 });
      if (noisy) {
        this._pauseNoiseMs += f.dt;
        if (this._pauseNoiseMs >= 120 && this._pauseNoiseMs <= 900 && !this._lastBreathCandidate) {
          this._lastBreathCandidate = true;
          this.breaths.push(f.t);
        }
      }
    }
    trimWindow(this.pauseNoise, f.t, 30000);
    this.onsets = this.onsets.filter((t) => t > f.t - 30000);
    this.breaths = this.breaths.filter((t) => t > f.t - 30000);
  }

  get speechSeconds() {
    return this.speechMs.reduce((a, s) => a + s.v, 0) / 1000;
  }

  evaluate() {
    const out = [];

    // PROSODY_FLAT: pitch spread in semitones. Conversational speech usually spans 2 to 4+ semitones of robust spread.
    if (this.f0.length >= 100) {
      const spread = robustStd(this.f0.map((p) => p.v));
      const fired = spread < 1.1;
      out.push({
        id: 'PROSODY_FLAT',
        fired,
        severity: fired ? clamp(0.9 - spread * 0.5, 0.35, 0.8) : 0,
        detail: `Pitch variation ${spread.toFixed(1)} semitones${fired ? ': monotone delivery, typical of synthesised speech.' : '.'}`,
      });
    }

    // VOCODER_ARTIFACT: spectral cliff above 5 kHz on most voiced frames.
    if (this.cliff.length >= 50) {
      const frac = this.cliff.reduce((a, c) => a + c.v, 0) / this.cliff.length;
      const fired = frac > 0.8;
      out.push({
        id: 'VOCODER_ARTIFACT',
        fired,
        severity: fired ? 0.5 : 0,
        detail: fired
          ? 'Hard spectral cutoff above 5 kHz on most voiced audio, a band-limited vocoder or codec signature. Some wideband codecs also do this.'
          : 'No abnormal spectral cutoff.',
      });
    }

    // BREATH_ABSENT: speech with pauses but no breath noise before phrase onsets.
    if (this.speechSeconds >= 15 && this.onsets.length >= 4) {
      const coupled = this.onsets.filter((o) => this.breaths.some((b) => o - b >= 0 && o - b < 900)).length;
      const coupling = coupled / this.onsets.length;
      const silent = this.pauseNoise.length ? this.pauseNoise.filter((p) => p.v).length / this.pauseNoise.length : 0;
      const fired = coupling < 0.1;
      out.push({
        id: 'BREATH_ABSENT',
        fired,
        severity: fired ? (silent > 0.8 ? 0.7 : 0.45) : 0,
        detail: fired
          ? `No breaths before ${this.onsets.length} phrases${silent > 0.8 ? ' and pauses are digitally silent' : ''}. Noise suppression can cause this too.`
          : `Breaths precede ${(coupling * 100).toFixed(0)}% of phrases.`,
      });
    }
    return out;
  }
}
