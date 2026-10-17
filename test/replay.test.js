import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { sessionFor } from '../server/analysis.js';
import { RecordingWriter, decodeAudioChunk, encodeAudioChunk, readRecording, safeLabel } from '../server/recording.js';
import { replay, summarize } from '../server/replay.js';
import { packFace } from '../public/js/wire.js';
import { LM_IDS } from '../public/js/wire.js';

const SR = 32000;

function tmpdir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'astraea-replay-'));
}

function env(seconds) {
  let s = 3;
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const bursts = [];
  for (let t = 0.2; t < seconds + 2; ) {
    const on = 0.15 + rand() * 0.25;
    bursts.push([t, on, 0.5 + rand() * 0.5]);
    t += on + 0.1 + rand() * 0.25;
  }
  return (t) => {
    for (const [t0, on, amp] of bursts) if (t >= t0 && t < t0 + on) return amp * Math.sin((Math.PI * (t - t0)) / on);
    return 0;
  };
}

const voice = (t, e) => e(t) * 0.3 * (Math.sin(2 * Math.PI * 140 * t) + 0.5 * Math.sin(2 * Math.PI * 280 * t) + 0.3 * Math.sin(2 * Math.PI * 420 * t));

function face(mouth) {
  const lm = [];
  for (const id of LM_IDS) lm[id] = { x: 0.5 + id * 1e-3, y: 0.5 + id * 1e-3 };
  lm[10] = { x: 0.5, y: 0.2 };
  lm[152] = { x: 0.5, y: 0.8 };
  lm[13] = { x: 0.5, y: 0.5 };
  lm[14] = { x: 0.5, y: 0.5 + mouth * 0.6 };
  return lm;
}

// Writes a recording the way the live socket does, and drives a session with the same data at the same time,
// so the test can assert that replaying the file reproduces what the live session saw.
async function record(file, { seconds, lagSec, label }) {
  const w = new RecordingWriter(file, { sampleRate: SR, label });
  const live = sessionFor(SR);
  const e = env(seconds);
  const chunk = 640;
  const clock = 5000;
  let nextFace = 0;
  for (let i = 0; i < seconds * SR; i += chunk) {
    const samples = new Float32Array(chunk).map((_, k) => voice((i + k) / SR - lagSec, e));
    const endT = clock + ((i + chunk) / SR) * 1000;
    const buf = encodeAudioChunk(samples, endT);
    const decoded = decodeAudioChunk(buf);
    live.pushAudio(decoded.samples, decoded.endT);
    w.audio(buf);
    while (nextFace <= endT) {
      const p = packFace(face(0.02 + 0.1 * e((nextFace - clock) / 1000)), null);
      live.pushFace(nextFace, p);
      w.face(nextFace, p);
      nextFace += 40;
    }
  }
  await w.close();
  return live;
}

test('audio chunks survive the recording round trip', () => {
  const x = new Float32Array([0, 0.5, -0.5, 1, -1]);
  const r = decodeAudioChunk(encodeAudioChunk(x, 1234.5));
  assert.equal(r.endT, 1234.5);
  for (let i = 0; i < x.length; i++) assert.ok(Math.abs(r.samples[i] - x[i]) < 1e-4);
  assert.equal(decodeAudioChunk(Buffer.alloc(4)), null);
});

test('labels cannot escape the recordings directory', () => {
  assert.equal(safeLabel('../../etc/passwd'), 'etcpasswd');
  assert.equal(safeLabel(''), 'capture');
  assert.equal(safeLabel(null), 'capture');
  assert.equal(safeLabel('Genuine Call #2'), 'genuinecall2');
});

test('replay reproduces what the live session measured', async () => {
  const dir = tmpdir();
  try {
    const file = path.join(dir, 'a.ndjson');
    const live = await record(file, { seconds: 12, lagSec: 0.3, label: 'fake' });
    const r = await replay(file);

    assert.equal(r.header.label, 'fake');
    assert.equal(r.header.sampleRate, SR);
    assert.ok(r.seconds >= 10, `${r.seconds} s replayed`);
    assert.equal(r.frames, live.frames);

    // The lag the live session measured must come back from the file. The live session is evaluated after the
    // last sample while the replay's final tick sits on a one-second boundary, so the correlation at those two
    // instants can differ slightly; the measured lag is the invariant.
    const lagOf = (detail) => Number(/by (\d+) ms/.exec(detail)[1]);
    const liveLip = live.evaluate().find((x) => x.id === 'LIP_DRIFT');
    const replayLip = r.signals.find((x) => x.id === 'LIP_DRIFT');
    assert.ok(replayLip.fired > 0, 'LIP_DRIFT fired during replay');
    assert.equal(lagOf(replayLip.lastDetail), lagOf(liveLip.detail));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('replay is deterministic', async () => {
  const dir = tmpdir();
  try {
    const file = path.join(dir, 'b.ndjson');
    await record(file, { seconds: 10, lagSec: 0.3, label: 'fake' });
    const a = await replay(file);
    const b = await replay(file);
    assert.deepEqual(a.ticks, b.ticks);
    assert.deepEqual(a.probes, b.probes);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('summary separates genuine from fake captures', async () => {
  const dir = tmpdir();
  try {
    const bad = path.join(dir, 'fake.ndjson');
    const good = path.join(dir, 'genuine.ndjson');
    await record(bad, { seconds: 12, lagSec: 0.3, label: 'fake' });
    await record(good, { seconds: 12, lagSec: 0, label: 'genuine' });
    const rows = Object.fromEntries(summarize([await replay(bad), await replay(good)]).map((r) => [r.id, r]));
    assert.equal(rows.LIP_DRIFT.genuine.fired, 0, 'no false positive on the in-sync capture');
    assert.ok(rows.LIP_DRIFT.fake.fired > 0, 'fires on the out-of-sync capture');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a corrupt recording is rejected rather than silently replayed', async () => {
  const dir = tmpdir();
  try {
    const file = path.join(dir, 'bad.ndjson');
    fs.writeFileSync(file, '{"type":"audio","b64":""}\n');
    await assert.rejects(() => replay(file), /not a header/);
    fs.writeFileSync(file, '{"type":"header","version":99}\n');
    await assert.rejects(() => replay(file), /unsupported recording version/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('readRecording yields the header first', async () => {
  const dir = tmpdir();
  try {
    const file = path.join(dir, 'c.ndjson');
    const w = new RecordingWriter(file, { sampleRate: SR, label: 'genuine' });
    w.face(1, packFace(face(0.1), null));
    await w.close();
    const types = [];
    for await (const rec of readRecording(file)) types.push(rec.type);
    assert.deepEqual(types, ['header', 'face']);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
