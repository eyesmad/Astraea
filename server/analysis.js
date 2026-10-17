// Server-side analysis session: one per capture. The browser streams 32 kHz PCM (stamped on its own
// performance.now() clock) and compact face frames; this runs the same metric classes the browser used to run.
import { brickwallDrop, estimateF0, rms, spectralFlatness } from '../public/js/audioFeatures.js';
import { AudioMetrics } from '../public/js/audioMetrics.js';
import { CrossModal } from '../public/js/crossmodal.js';
import { VideoMetrics, mouthAperture } from '../public/js/videoMetrics.js';
import { unpackFace } from '../public/js/wire.js';
import { decimate2, spectrumDb } from './dsp.js';
import { MODEL_SR, sharedVoiceModel } from './voiceModel.js';

// The model reads a 4 s window. Scoring is slower than real time on CPU, so windows are scored at an
// interval rather than continuously, and a run of agreeing scores is required before the signal fires:
// a single window is noisy, and this signal is weighted heavily enough that a lone spike would mislead.
export const MODEL_WINDOW_SEC = 4;
export const MODEL_INTERVAL_MS = 3000;
export const MODEL_MIN_SCORES = 3;
export const MODEL_THRESHOLD = 0.85;
export const MODEL_SPEECH_RMS = 0.02;
// The model is confident on anything that is not speech: a sine tone and white noise both score over 99%
// synthetic. Scoring only windows containing real voiced audio keeps hold music and line noise out of it.
export const MODEL_MIN_VOICED = 25;

export const SAMPLE_RATES = new Set([16000, 32000, 44100, 48000]);

// Single place that maps a capture rate to session settings, so a replay reconstructs exactly the session
// the live socket built for that recording.
export function sessionFor(sampleRate, { voiceModel = sharedVoiceModel() } = {}) {
  const sr = SAMPLE_RATES.has(sampleRate) ? sampleRate : 32000;
  return new AnalysisSession({ sampleRate: sr, fftSize: sr > 24000 ? 2048 : 1024, voiceModel });
}

export class AnalysisSession {
  constructor({ sampleRate = 32000, fftSize = 2048, hopMs = 20, voiceModel = null } = {}) {
    this.voiceModel = voiceModel;
    this.sr = sampleRate;
    this.fftSize = fftSize;
    this.hop = Math.round((sampleRate * hopMs) / 1000);
    this.hopMs = (this.hop / sampleRate) * 1000;
    this.audio = new AudioMetrics();
    this.video = new VideoMetrics();
    this.cross = new CrossModal();
    this.reset();
  }

  reset() {
    this.audio.reset();
    this.video.reset();
    this.cross.reset();
    this.ring = new Float32Array(this.fftSize);
    this.filled = 0; // samples written to the ring, capped at fftSize
    this.sinceHop = 0;
    this.total = 0; // samples received
    this.offset = null; // client ms at sample 0, tracked as a slowly rising lower envelope (jitter only adds delay)
    this.now = 0;
    this.tick = 0;
    this.f0 = { f0: 0, clarity: 0 };
    this.drop = { dropDb: 0 };
    this.lastFrameT = 0;
    this.modelRing = new Float32Array(this.sr * MODEL_WINDOW_SEC);
    this.modelFilled = 0;
    this.modelScores = []; // { t, score }
    this.modelInFlight = null;
    this.lastScoreT = 0;
  }

  // samples: Float32Array in [-1, 1]; endT: client performance.now() when the last sample was captured.
  pushAudio(samples, endT) {
    const endIdx = this.total + samples.length;
    const off = endT - (endIdx / this.sr) * 1000;
    this.offset = this.offset == null ? off : Math.min(off, this.offset + 0.02);
    for (let i = 0; i < samples.length; i++) {
      this.ring[this.total % this.fftSize] = samples[i];
      this.modelRing[this.total % this.modelRing.length] = samples[i];
      if (this.modelFilled < this.modelRing.length) this.modelFilled++;
      this.total++;
      if (this.filled < this.fftSize) this.filled++;
      if (++this.sinceHop >= this.hop) {
        this.sinceHop = 0;
        if (this.filled === this.fftSize) this._frame(this.offset + (this.total / this.sr) * 1000);
      }
    }
  }

  _frame(t) {
    const n = this.fftSize;
    const frame = new Float32Array(n);
    const start = this.total % n;
    for (let i = 0; i < n; i++) frame[i] = this.ring[(start + i) % n];
    const level = rms(frame);
    this.cross.pushAudio(t, level);
    this.now = Math.max(this.now, t);

    const freq = spectrumDb(frame);
    const hz = this.sr / n;
    const flat = spectralFlatness(freq, Math.floor(300 / hz), Math.floor(4000 / hz));
    if (level >= 0.02 && this.tick++ % 2 === 0) {
      this.f0 = estimateF0(frame, this.sr);
      this.drop = brickwallDrop(freq, this.sr, n);
    }
    const dt = this.lastFrameT ? t - this.lastFrameT : this.hopMs;
    this.lastFrameT = t;
    this.audio.push({ t, dt, rms: level, flatness: flat, f0: this.f0.f0, clarity: this.f0.clarity, dropDb: this.drop.dropDb });
    this._maybeScore(t, level);
  }

  // Fire-and-forget: scoring takes seconds, so the result lands in a later evaluate() rather than blocking
  // the audio path. Windows that arrive while the model is busy are dropped, never queued.
  _maybeScore(t, level) {
    const model = this.voiceModel;
    if (!model || model.busy) return;
    if (this.modelFilled < this.modelRing.length) return;
    if (level < MODEL_SPEECH_RMS) return; // silence scores as noise
    if (this.lastScoreT && t - this.lastScoreT < MODEL_INTERVAL_MS) return;
    const windowStart = t - MODEL_WINDOW_SEC * 1000;
    if (this.audio.f0.filter((p) => p.t >= windowStart).length < MODEL_MIN_VOICED) return;
    this.lastScoreT = t;

    const n = this.modelRing.length;
    const window = new Float32Array(n);
    const start = this.total % n;
    for (let i = 0; i < n; i++) window[i] = this.modelRing[(start + i) % n];
    let pcm = window;
    for (let rate = this.sr; rate > MODEL_SR; rate /= 2) pcm = decimate2(pcm);

    this.modelInFlight = model.score(pcm).then((score) => {
      if (typeof score === 'number') {
        this.modelScores.push({ t, score });
        if (this.modelScores.length > 8) this.modelScores.shift();
      }
    }).catch(() => {});
  }

  // Live capture never waits for the model, but a replay must: otherwise the run would finish before the
  // scores land and the same recording would score differently from one run to the next.
  async drainModel() {
    while (this.modelInFlight) {
      const p = this.modelInFlight;
      await p;
      if (this.modelInFlight === p) this.modelInFlight = null;
    }
  }

  // Median of the recent scores: one odd window should not move the signal.
  modelVerdict() {
    if (this.modelScores.length < MODEL_MIN_SCORES) return null;
    const v = this.modelScores.map((s) => s.score).sort((a, b) => a - b);
    return { median: v[Math.floor(v.length / 2)], n: v.length };
  }

  evaluateModel() {
    const v = this.modelVerdict();
    if (!v) return [];
    const fired = v.median >= MODEL_THRESHOLD;
    return [{
      id: 'VOICE_MODEL',
      fired,
      severity: fired ? Math.min(1, 0.5 + (v.median - MODEL_THRESHOLD) * 2) : 0,
      detail: `Voice model scores this ${(v.median * 100).toFixed(0)}% synthetic over ${v.n} windows${fired ? '.' : ', below the alert threshold.'}`,
    }];
  }

  pushFace(t, packed) {
    const face = unpackFace(packed);
    if (!face) return false;
    this.video.push(t, face.lm, face.blend);
    this.cross.pushMouth(t, mouthAperture(face.lm));
    this.now = Math.max(this.now, t);
    return true;
  }

  evaluate() {
    return [...this.video.evaluate(), ...this.audio.evaluate(), ...this.cross.evaluate(this.now), ...this.evaluateModel()];
  }

  // Live numeric read-outs grouped by layer, for the overlay's "what is being analysed" panel.
  probes() {
    const voice = this.audio.probe();
    if (this.voiceModel) {
      const v = this.modelVerdict();
      voice.push({
        key: 'model',
        label: 'Voice model',
        progress: Math.min(1, this.modelScores.length / MODEL_MIN_SCORES),
        text: v ? `${(v.median * 100).toFixed(0)}% synthetic over ${v.n} windows` : 'warming up',
        bar: v ? Math.min(1, Math.max(0, v.median)) : 0,
      });
    }
    return { video: this.video.probe(), voice, crossmodal: this.cross.probe(this.now) };
  }

  get frames() {
    return this.video.frames;
  }
}
