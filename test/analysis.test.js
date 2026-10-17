import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import WebSocket from 'ws';
import { spectrumDb, decimate2 } from '../server/dsp.js';
import { AnalysisSession } from '../server/analysis.js';
import { attachAnalysis } from '../server/analysisSocket.js';
import { LM_IDS, packFace, unpackFace } from '../public/js/wire.js';
import { BLEND_KEYS } from '../public/js/videoMetrics.js';

const SR = 32000;

// Syllable-like envelope in [0, 1] as a function of time in seconds (deterministic).
function makeEnv(seconds) {
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

const voice = (t, env) => env(t) * 0.3 * (Math.sin(2 * Math.PI * 140 * t) + 0.5 * Math.sin(2 * Math.PI * 280 * t) + 0.3 * Math.sin(2 * Math.PI * 420 * t));

function face(mouth) {
  const lm = [];
  for (const id of LM_IDS) lm[id] = { x: 0.5 + id * 1e-3, y: 0.5 + id * 1e-3 };
  lm[10] = { x: 0.5, y: 0.2 };
  lm[152] = { x: 0.5, y: 0.8 };
  lm[13] = { x: 0.5, y: 0.5 };
  lm[14] = { x: 0.5, y: 0.5 + mouth * 0.6 };
  return lm;
}

test('spectrumDb puts a pure tone in the right bin', () => {
  const n = 2048;
  const x = new Float32Array(n).map((_, i) => 0.5 * Math.sin((2 * Math.PI * 1000 * i) / SR));
  const db = spectrumDb(x);
  const peak = db.indexOf(Math.max(...db));
  assert.ok(Math.abs(peak - Math.round((1000 * n) / SR)) <= 1);
});

test('decimate2 keeps a 1 kHz tone and halves the length', () => {
  const x = new Float32Array(3200).map((_, i) => Math.sin((2 * Math.PI * 1000 * i) / SR));
  const y = decimate2(x);
  assert.equal(y.length, 1600);
  const mid = y.slice(100, 1500);
  assert.ok(Math.max(...mid) > 0.95 && Math.max(...mid) < 1.05);
});

test('face packing round-trips the landmarks the metrics read', () => {
  const blend = Object.fromEntries(BLEND_KEYS.map((k, i) => [k, i / 20]));
  const out = unpackFace(packFace(face(0.1), blend));
  assert.ok(Math.abs(out.lm[14].y - 0.56) < 1e-4);
  assert.equal(out.blend.jawOpen, 0);
  assert.equal(unpackFace(packFace(face(0.1), null)).blend, null);
  assert.equal(unpackFace([1, 2, 3]), null);
  assert.equal(unpackFace(null), null);
});

function drive(session, seconds, lagSec) {
  const env = makeEnv(seconds);
  const clockOffset = 5000; // client clock is not zero-based
  const chunk = 640;
  let nextFace = 0;
  for (let i = 0; i < seconds * SR; i += chunk) {
    const x = new Float32Array(chunk).map((_, k) => voice((i + k) / SR - lagSec, env));
    const endT = clockOffset + ((i + chunk) / SR) * 1000;
    session.pushAudio(x, endT);
    while (nextFace <= endT) {
      session.pushFace(nextFace, packFace(face(0.02 + 0.1 * env((nextFace - clockOffset) / 1000)), null));
      nextFace += 40;
    }
  }
}

test('server session recovers a 300 ms audio lag and fires LIP_DRIFT', () => {
  const s = new AnalysisSession({ sampleRate: SR });
  drive(s, 12, 0.3);
  const lip = s.evaluate().find((r) => r.id === 'LIP_DRIFT');
  assert.ok(lip, 'cross-modal result present');
  assert.equal(lip.fired, true);
  const ms = Number(/by (\d+) ms/.exec(lip.detail)[1]);
  assert.ok(Math.abs(ms - 300) <= 80, lip.detail);
});

test('server session sees aligned audio as in sync', () => {
  const s = new AnalysisSession({ sampleRate: SR });
  drive(s, 12, 0);
  const lip = s.evaluate().find((r) => r.id === 'LIP_DRIFT');
  assert.ok(lip);
  assert.equal(lip.fired, false, lip.detail);
});

test('analysis socket streams signals back and rejects foreign origins', async () => {
  const server = http.createServer();
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address();
  attachAnalysis(server, port);
  try {
    const ws = new WebSocket(`ws://127.0.0.1:${port}/ws/analyze`, { origin: `http://127.0.0.1:${port}` });
    await new Promise((res, rej) => { ws.on('open', res); ws.on('error', rej); });
    ws.send(JSON.stringify({ type: 'config', sampleRate: SR }));
    const got = new Promise((res) => ws.on('message', (d) => { const m = JSON.parse(d); if (m.type === 'signals') res(m); }));
    const env = makeEnv(3);
    for (let i = 0; i < 2 * SR; i += 640) {
      const buf = Buffer.alloc(8 + 640 * 2);
      buf.writeDoubleLE(1000 + ((i + 640) / SR) * 1000, 0);
      for (let k = 0; k < 640; k++) buf.writeInt16LE(Math.round(voice((i + k) / SR, env) * 32767), 8 + 2 * k);
      ws.send(buf);
    }
    ws.send(JSON.stringify({ type: 'face', t: 1500, p: packFace(face(0.05), null) }));
    const m = await got;
    assert.ok(Array.isArray(m.results));
    assert.equal(m.frames, 1);
    ws.close();

    const bad = new WebSocket(`ws://127.0.0.1:${port}/ws/analyze`, { origin: 'http://evil.example' });
    const status = await new Promise((res) => { bad.on('unexpected-response', (_q, r) => res(r.statusCode)); bad.on('error', () => {}); });
    assert.equal(status, 403);
  } finally {
    server.closeAllConnections?.();
    server.close();
  }
});
