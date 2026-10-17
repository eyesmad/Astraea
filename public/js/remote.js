// Client for the server-side analysis socket: streams PCM and face frames up, receives signals back.
import { packFace } from './wire.js';

const SAMPLE_RATE = 32000;

// ?record=genuine or ?record=fake asks the server to save the capture for later replay. Labelled captures
// are what the thresholds get calibrated against, so the label matters more than the filename.
export function recordLabel() {
  const v = new URLSearchParams(location.search).get('record');
  return v === '' ? 'capture' : v;
}

export class RemoteAnalysis {
  constructor({ onSignals, onClose } = {}) {
    this.onSignals = onSignals;
    this.onClose = onClose;
    this.ws = null;
    this.ctx = null;
    this.frames = 0;
    this.probes = null;
    this.recording = false;
  }

  get active() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect() {
    return new Promise((resolve) => {
      let settled = false;
      const done = (ok) => { if (!settled) { settled = true; resolve(ok); } };
      try {
        const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/analyze`);
        this.ws = ws;
        const timeout = setTimeout(() => { done(false); ws.close(); }, 3000);
        ws.onopen = () => {
          clearTimeout(timeout);
          const record = recordLabel();
          ws.send(JSON.stringify({ type: 'config', sampleRate: SAMPLE_RATE, ...(record ? { record } : {}) }));
          done(true);
        };
        ws.onmessage = (e) => {
          const m = JSON.parse(e.data);
          if (m.type === 'ready') {
            this.recording = Boolean(m.recording);
          } else if (m.type === 'signals') {
            this.frames = m.frames;
            this.probes = m.probes;
            this.onSignals?.(m.results);
          }
        };
        ws.onclose = () => { clearTimeout(timeout); done(false); this.onClose?.(); };
        ws.onerror = () => done(false);
      } catch {
        done(false);
      }
    });
  }

  async startAudio(stream) {
    if (!stream.getAudioTracks().length || !this.active) return false;
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    await this.ctx.resume();
    await this.ctx.audioWorklet.addModule('js/pcm-worklet.js');
    const src = this.ctx.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
    const tap = new AudioWorkletNode(this.ctx, 'pcm-tap');
    tap.port.onmessage = (e) => this._sendAudio(e.data);
    const mute = this.ctx.createGain(); // a worklet is only pulled when it reaches the destination; muted so nothing plays
    mute.gain.value = 0;
    src.connect(tap).connect(mute).connect(this.ctx.destination);
    return true;
  }

  _sendAudio(block) {
    if (!this.active || this.ws.bufferedAmount > 1 << 20) return;
    const buf = new ArrayBuffer(8 + block.length * 2);
    const dv = new DataView(buf);
    dv.setFloat64(0, performance.now(), true);
    for (let i = 0; i < block.length; i++) dv.setInt16(8 + 2 * i, Math.max(-1, Math.min(1, block[i])) * 32767, true);
    this.ws.send(buf);
  }

  sendFace(t, lm, blend) {
    if (this.active) this.ws.send(JSON.stringify({ type: 'face', t, p: packFace(lm, blend) }));
  }

  reset() {
    if (this.active) this.ws.send(JSON.stringify({ type: 'reset' }));
  }

  stop() {
    this.ctx?.close();
    this.ctx = null;
    const ws = this.ws;
    this.ws = null;
    if (ws) { ws.onclose = null; ws.close(); }
    this.frames = 0;
    this.probes = null;
    this.recording = false;
  }
}
