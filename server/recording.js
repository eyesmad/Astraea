// Capture recordings: the exact stream the browser sends to the analysis socket, written to disk so a real
// call can be replayed through the pipeline deterministically. Recording at the wire level means a replay
// needs no camera, no MediaPipe and no video file, and always produces the same result.
//
// Format: newline-delimited JSON. First line is a header, then audio and face records in arrival order.
// Audio is kept as the original binary chunk (base64), so replay parses exactly what the socket parsed.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';

export const RECORDING_VERSION = 1;
export const RECORDING_DIR = 'recordings';

// A chunk is an 8-byte little-endian client timestamp (performance.now() at the last sample) then int16 PCM.
export function decodeAudioChunk(buf) {
  const n = (buf.length - 8) >> 1;
  if (n <= 0) return null;
  const endT = buf.readDoubleLE(0);
  if (!Number.isFinite(endT)) return null;
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) samples[i] = buf.readInt16LE(8 + 2 * i) / 32768;
  return { samples, endT };
}

export function encodeAudioChunk(samples, endT) {
  const buf = Buffer.alloc(8 + samples.length * 2);
  buf.writeDoubleLE(endT, 0);
  for (let i = 0; i < samples.length; i++) {
    buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), 8 + 2 * i);
  }
  return buf;
}

// Labels become part of a filename and arrive from the browser, so allow only a safe slug.
export function safeLabel(label) {
  const s = String(label ?? '').toLowerCase().replace(/[^a-z0-9_-]/g, '').slice(0, 40);
  return s || 'capture';
}

export class RecordingWriter {
  constructor(file, header) {
    this.file = file;
    this.audioChunks = 0;
    this.faces = 0;
    fs.mkdirSync(path.dirname(file), { recursive: true });
    this.stream = fs.createWriteStream(file);
    this._write({ type: 'header', version: RECORDING_VERSION, startedAt: new Date().toISOString(), ...header });
  }

  _write(obj) {
    this.stream.write(`${JSON.stringify(obj)}\n`);
  }

  audio(buf) {
    this.audioChunks++;
    this._write({ type: 'audio', b64: buf.toString('base64') });
  }

  face(t, p) {
    this.faces++;
    this._write({ type: 'face', t, p });
  }

  close() {
    return new Promise((resolve) => this.stream.end(resolve));
  }
}

export function recordingPath(label, dir = RECORDING_DIR) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  return path.join(dir, `${safeLabel(label)}-${stamp}.ndjson`);
}

// Yields the header first, then each record in order.
export async function* readRecording(file) {
  const input = fs.createReadStream(file);
  const rl = readline.createInterface({ input, crlfDelay: Infinity });
  let header = null;
  try {
    for await (const line of rl) {
      if (!line.trim()) continue;
      const rec = JSON.parse(line);
      if (!header) {
        if (rec.type !== 'header') throw new Error(`${file}: first line is not a header`);
        if (rec.version !== RECORDING_VERSION) throw new Error(`${file}: unsupported recording version ${rec.version}`);
        header = rec;
      }
      yield rec;
    }
    if (!header) throw new Error(`${file}: empty recording`);
  } finally {
    // Runs when the caller stops early or a line is rejected, so the file is never left open. The close is
    // awaited because the handle outlives destroy() by a tick, and on Windows that blocks deleting the file.
    rl.close();
    await new Promise((resolve) => input.close(resolve));
  }
}
