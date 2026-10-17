// Optional voice-deepfake model, run as a Python child process (see tools/voice_model_sidecar.py).
// The model is a bonus signal, never a dependency: if Python is missing, the weights cannot be fetched, or
// the process dies, scoring just stops and the heuristics carry the voice layer on their own.
import { spawn } from 'node:child_process';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCRIPT = path.join(__dirname, '..', 'tools', 'voice_model_sidecar.py');

export const MODEL_SR = 16000;
export const STARTUP_TIMEOUT_MS = 900000; // first run downloads about 1.2 GB of weights, then loads it
export const SCORE_TIMEOUT_MS = 30000;

export function voiceModelEnabled(env = process.env) {
  return Boolean(env.VOICE_MODEL) && env.VOICE_MODEL !== '0';
}

export class VoiceModel {
  constructor({ env = process.env, script = SCRIPT } = {}) {
    this.python = env.VOICE_MODEL_PYTHON || 'python';
    this.script = script;
    this.env = env;
    this.proc = null;
    this.ready = null;
    this.pending = null; // one in flight at a time; windows are dropped rather than queued
    this.failed = null;
  }

  get busy() {
    return this.pending !== null;
  }

  start() {
    if (this.ready) return this.ready;
    this.ready = new Promise((resolve) => {
      let settled = false;
      const finish = (ok, why) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (!ok) {
          this.failed = why;
          this.stop();
        }
        resolve(ok);
      };
      const timer = setTimeout(() => finish(false, 'timed out starting (model download can take a while)'), STARTUP_TIMEOUT_MS);

      try {
        this.proc = spawn(this.python, [this.script], { env: this.env, stdio: ['pipe', 'pipe', 'pipe'] });
      } catch (err) {
        finish(false, err.message);
        return;
      }
      this.proc.on('error', (err) => {
        this.rejectPending(err.message);
        finish(false, err.message);
      });
      this.proc.on('exit', (code) => {
        this.rejectPending(`sidecar exited (${code})`);
        finish(false, `sidecar exited (${code})`);
        this.proc = null;
      });
      this.proc.stderr.on('data', (d) => {
        const line = d.toString().trim();
        if (line && line !== 'ready') console.error(`[voice-model] ${line}`);
      });
      readline.createInterface({ input: this.proc.stdout }).on('line', (line) => {
        let msg;
        try {
          msg = JSON.parse(line);
        } catch {
          return;
        }
        if (msg.ready) {
          finish(true);
        } else if (this.pending) {
          const { resolve: res, timer: t } = this.pending;
          this.pending = null;
          clearTimeout(t);
          res(typeof msg.score === 'number' ? msg.score : null);
        }
      });
    });
    return this.ready;
  }

  rejectPending(why) {
    if (!this.pending) return;
    const { resolve: res, timer } = this.pending;
    this.pending = null;
    clearTimeout(timer);
    console.error(`[voice-model] ${why}`);
    res(null);
  }

  // Returns a 0-1 score (higher means more likely synthetic), or null when the window could not be scored.
  async score(samples) {
    if (!(await this.start())) return null;
    if (this.busy || !this.proc?.stdin.writable) return null;
    return new Promise((resolve) => {
      const timer = setTimeout(() => this.rejectPending('scoring timed out'), SCORE_TIMEOUT_MS);
      this.pending = { resolve, timer };
      const header = Buffer.alloc(4);
      header.writeUInt32LE(samples.length, 0);
      const body = Buffer.alloc(samples.length * 4);
      for (let i = 0; i < samples.length; i++) body.writeFloatLE(samples[i], i * 4);
      this.proc.stdin.write(Buffer.concat([header, body]));
    });
  }

  stop() {
    this.rejectPending('stopped');
    this.proc?.kill();
    this.proc = null;
  }
}

let shared = null;

// One process for the whole server: loading the weights takes far longer than scoring a window.
export function sharedVoiceModel(env = process.env) {
  if (!voiceModelEnabled(env)) return null;
  if (!shared) shared = new VoiceModel({ env });
  return shared;
}
