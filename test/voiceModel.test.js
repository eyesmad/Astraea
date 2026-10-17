import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { AnalysisSession, MODEL_MIN_SCORES, MODEL_THRESHOLD } from '../server/analysis.js';
import { VoiceModel, voiceModelEnabled } from '../server/voiceModel.js';

const SR = 32000;

// A stand-in for the Python sidecar: same length-prefixed protocol, no torch required.
function stubSidecar(dir, { score = 0.95, crash = false, silent = false } = {}) {
  const file = path.join(dir, 'stub.js');
  fs.writeFileSync(file, [
    `if (${crash}) { console.error('boom'); process.exit(3); }`,
    `if (!${silent}) console.log(JSON.stringify({ ready: true }));`,
    'let buf = Buffer.alloc(0);',
    "process.stdin.on('data', (d) => {",
    '  buf = Buffer.concat([buf, d]);',
    '  while (buf.length >= 4) {',
    '    const n = buf.readUInt32LE(0);',
    '    if (buf.length < 4 + n * 4) break;',
    '    buf = buf.subarray(4 + n * 4);',
    `    console.log(JSON.stringify({ score: ${score} }));`,
    '  }',
    '});',
  ].join('\n'));
  return file;
}

function modelWith(dir, opts) {
  return new VoiceModel({ script: stubSidecar(dir, opts), env: { ...process.env, VOICE_MODEL_PYTHON: process.execPath } });
}

function tmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'astraea-vm-'));
}

const speech = (t) => 0.3 * (Math.sin(2 * Math.PI * 140 * t) + 0.5 * Math.sin(2 * Math.PI * 280 * t));

async function drive(session, seconds) {
  const chunk = 640;
  for (let i = 0; i < seconds * SR; i += chunk) {
    const x = new Float32Array(chunk).map((_, k) => speech((i + k) / SR));
    session.pushAudio(x, 5000 + ((i + chunk) / SR) * 1000);
    await session.drainModel();
  }
}

test('voiceModelEnabled respects the env flag', () => {
  assert.equal(voiceModelEnabled({}), false);
  assert.equal(voiceModelEnabled({ VOICE_MODEL: '0' }), false);
  assert.equal(voiceModelEnabled({ VOICE_MODEL: '1' }), true);
});

test('scores a window over the sidecar protocol', async () => {
  const dir = tmp();
  const m = modelWith(dir);
  try {
    assert.equal(await m.score(new Float32Array(64000).fill(0.1)), 0.95);
  } finally {
    m.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a sidecar that cannot start degrades to no score', async () => {
  const dir = tmp();
  const m = modelWith(dir, { crash: true });
  try {
    assert.equal(await m.score(new Float32Array(64000)), null);
    assert.ok(m.failed, 'records why it failed');
  } finally {
    m.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a missing python binary degrades to no score', async () => {
  const m = new VoiceModel({ script: 'nope.py', env: { VOICE_MODEL_PYTHON: 'definitely-not-a-binary-xyz' } });
  try {
    assert.equal(await m.score(new Float32Array(16000)), null);
  } finally {
    m.stop();
  }
});

test('the session fires VOICE_MODEL only after several agreeing windows', async () => {
  const dir = tmp();
  const m = modelWith(dir, { score: 0.97 });
  const s = new AnalysisSession({ sampleRate: SR, voiceModel: m });
  try {
    await drive(s, 12);
    assert.ok(s.modelScores.length >= MODEL_MIN_SCORES, `${s.modelScores.length} windows scored`);
    const r = s.evaluate().find((x) => x.id === 'VOICE_MODEL');
    assert.equal(r.fired, true, r.detail);
    assert.ok(r.detail.includes('97%'));

    // One window alone must not be enough to fire a signal this heavily weighted.
    const fresh = new AnalysisSession({ sampleRate: SR, voiceModel: m });
    fresh.modelScores.push({ t: 1, score: 0.99 });
    assert.deepEqual(fresh.evaluate().filter((x) => x.id === 'VOICE_MODEL'), []);
  } finally {
    m.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a confidently real voice does not fire', async () => {
  const dir = tmp();
  const m = modelWith(dir, { score: 0.04 });
  const s = new AnalysisSession({ sampleRate: SR, voiceModel: m });
  try {
    await drive(s, 12);
    const r = s.evaluate().find((x) => x.id === 'VOICE_MODEL');
    assert.equal(r.fired, false, r.detail);
    assert.ok(MODEL_THRESHOLD > 0.5, 'threshold is deliberately stricter than the model default');
  } finally {
    m.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('silence is never scored', async () => {
  const dir = tmp();
  const m = modelWith(dir);
  const s = new AnalysisSession({ sampleRate: SR, voiceModel: m });
  try {
    for (let i = 0; i < 8 * SR; i += 640) {
      s.pushAudio(new Float32Array(640), 5000 + ((i + 640) / SR) * 1000);
      await s.drainModel();
    }
    assert.equal(s.modelScores.length, 0);
    assert.deepEqual(s.evaluate().filter((x) => x.id === 'VOICE_MODEL'), []);
  } finally {
    m.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('without a model the session behaves exactly as before', async () => {
  const s = new AnalysisSession({ sampleRate: SR });
  await drive(s, 6);
  assert.deepEqual(s.evaluate().filter((x) => x.id === 'VOICE_MODEL'), []);
  assert.ok(!s.probes().voice.some((p) => p.key === 'model'));
});

test('non-speech audio is never scored', async () => {
  const dir = tmp();
  const m = modelWith(dir);
  const s = new AnalysisSession({ sampleRate: SR, voiceModel: m });
  try {
    // Loud white noise: above the speech level, but with no voiced pitch the model would call it synthetic.
    for (let i = 0; i < 12 * SR; i += 640) {
      s.pushAudio(new Float32Array(640).map(() => (Math.random() - 0.5) * 0.6), 5000 + ((i + 640) / SR) * 1000);
      await s.drainModel();
    }
    assert.equal(s.modelScores.length, 0, 'noise must not reach the model');
  } finally {
    m.stop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
