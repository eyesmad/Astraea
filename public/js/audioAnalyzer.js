// Web Audio wrapper: pulls frames from an AnalyserNode and feeds AudioMetrics and the cross-modal envelope.
import { brickwallDrop, estimateF0, rms, spectralFlatness } from './audioFeatures.js';
import { AudioMetrics } from './audioMetrics.js';

export class AudioAnalyzer {
  constructor({ onEnvelope } = {}) {
    this.metrics = new AudioMetrics();
    this.onEnvelope = onEnvelope;
    this.timer = null;
    this.ctx = null;
  }

  async start(stream) {
    if (!stream.getAudioTracks().length) return false;
    this.ctx = new AudioContext();
    await this.ctx.resume();
    const src = this.ctx.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0;
    this.analyser.minDecibels = -120;
    src.connect(this.analyser); // not connected to destination: analysis only, no playback
    this.time = new Float32Array(this.analyser.fftSize);
    this.freq = new Float32Array(this.analyser.frequencyBinCount);
    this.last = performance.now();
    this.tick = 0;
    this.f0 = { f0: 0, clarity: 0 };
    this.drop = { dropDb: 0 };
    // setInterval rather than rAF: rAF stops when the tab is hidden behind the meeting window.
    this.timer = setInterval(() => this._frame(), 20);
    return true;
  }

  _frame() {
    const t = performance.now();
    const dt = t - this.last;
    this.last = t;
    this.analyser.getFloatTimeDomainData(this.time);
    this.analyser.getFloatFrequencyData(this.freq);
    const level = rms(this.time);
    this.onEnvelope?.(t, level);

    const sr = this.ctx.sampleRate;
    const hz = sr / this.analyser.fftSize;
    const flat = spectralFlatness(this.freq, Math.floor(300 / hz), Math.floor(4000 / hz));
    // Pitch and spectral-cliff analysis are heavier, so run them on every 2nd frame for voiced audio.
    if (level >= 0.02 && this.tick++ % 2 === 0) {
      this.f0 = estimateF0(this.time, sr);
      this.drop = brickwallDrop(this.freq, sr, this.analyser.fftSize);
    }
    this.metrics.push({ t, dt, rms: level, flatness: flat, f0: this.f0.f0, clarity: this.f0.clarity, dropDb: this.drop.dropDb });
  }

  evaluate() {
    return this.metrics.evaluate();
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    this.ctx?.close();
    this.ctx = null;
    this.metrics.reset();
  }
}
