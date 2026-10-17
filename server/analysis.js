// Server-side analysis session: one per capture. The browser streams 32 kHz PCM (stamped on its own
// performance.now() clock) and compact face frames; this runs the same metric classes the browser used to run.
import { brickwallDrop, estimateF0, rms, spectralFlatness } from '../public/js/audioFeatures.js';
import { AudioMetrics } from '../public/js/audioMetrics.js';
import { CrossModal } from '../public/js/crossmodal.js';
import { VideoMetrics, mouthAperture } from '../public/js/videoMetrics.js';
import { unpackFace } from '../public/js/wire.js';
import { spectrumDb } from './dsp.js';

export class AnalysisSession {
  constructor({ sampleRate = 32000, fftSize = 2048, hopMs = 20 } = {}) {
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
  }

  // samples: Float32Array in [-1, 1]; endT: client performance.now() when the last sample was captured.
  pushAudio(samples, endT) {
    const endIdx = this.total + samples.length;
    const off = endT - (endIdx / this.sr) * 1000;
    this.offset = this.offset == null ? off : Math.min(off, this.offset + 0.02);
    for (let i = 0; i < samples.length; i++) {
      this.ring[this.total % this.fftSize] = samples[i];
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
    return [...this.video.evaluate(), ...this.audio.evaluate(), ...this.cross.evaluate(this.now)];
  }

  get frames() {
    return this.video.frames;
  }
}
