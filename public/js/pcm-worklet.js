// AudioWorklet: batches the 128-sample render quanta into 20 ms blocks and posts them to the main thread.
class PcmTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(Math.round(sampleRate * 0.02));
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.block[this.n++] = ch[i];
      if (this.n === this.block.length) {
        this.port.postMessage(this.block.slice());
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-tap', PcmTap);
