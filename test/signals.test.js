import test from 'node:test';
import assert from 'node:assert/strict';
import { crossCorrelate, resample, zscore } from '../public/js/crosscorr.js';
import { CrossModal } from '../public/js/crossmodal.js';
import { CompoundDetector, computeTrust, TrustLedger } from '../public/js/compound.js';
import { SIGNALS } from '../public/js/signals.js';
import { FaceTracker } from '../public/js/faceMarkers.js';
import { estimateF0, brickwallDrop, rms } from '../public/js/audioFeatures.js';
import { AudioMetrics } from '../public/js/audioMetrics.js';
import { VideoMetrics } from '../public/js/videoMetrics.js';
import { isThumbsUp, isPeace, isTouchingNose } from '../public/js/gestures.js';

// Speech-like envelope: bursts of varying length.
function envelope(n, hz, seed = 1) {
  let s = seed;
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const out = new Array(n).fill(0);
  let i = 0;
  while (i < n) {
    const on = Math.floor(hz * (0.2 + rand() * 0.4));
    const off = Math.floor(hz * (0.1 + rand() * 0.3));
    const amp = 0.4 + rand() * 0.6;
    for (let k = 0; k < on && i < n; k++, i++) out[i] = amp * Math.sin((Math.PI * k) / on);
    i += off;
  }
  return out;
}

test('crossCorrelate recovers a known lag', () => {
  const x = envelope(250, 25);
  const shift = 6; // audio trails mouth by 6 samples = 240 ms
  const y = x.map((_, i) => x[i - shift] ?? 0);
  const { lag, corr } = crossCorrelate(x, y, 12);
  assert.equal(lag, shift);
  assert.ok(corr > 0.9);
});

test('crossCorrelate finds zero lag for aligned tracks', () => {
  const x = envelope(250, 25, 7);
  assert.equal(crossCorrelate(x, x, 12).lag, 0);
});

test('resample interpolates onto a uniform grid', () => {
  const track = [{ t: 0, v: 0 }, { t: 1000, v: 10 }];
  const r = resample(track, 10, 0, 1000);
  assert.equal(r.length, 11);
  assert.ok(Math.abs(r[5] - 5) < 1e-9);
});

test('CrossModal flags offset audio but not synced audio', () => {
  const hz = 50;
  const seconds = 10;
  const mouthEnv = envelope(seconds * hz, hz, 3);
  const run = (shiftSamples) => {
    const cm = new CrossModal();
    for (let i = 0; i < mouthEnv.length; i++) {
      const t = i * (1000 / hz);
      cm.pushMouth(t, 0.02 + 0.1 * mouthEnv[i]);
      cm.pushAudio(t, 0.01 + 0.2 * (mouthEnv[i - shiftSamples] ?? 0));
    }
    return Object.fromEntries(cm.evaluate(seconds * 1000 - 1).map((r) => [r.id, r]));
  };
  assert.equal(run(0).LIP_DRIFT.fired, false);
  const drift = run(15); // 300 ms
  assert.equal(drift.LIP_DRIFT.fired, true);
  assert.ok(drift.LIP_DRIFT.detail.includes('lags'));
});

test('CompoundDetector needs two live signals inside the window', () => {
  const c = new CompoundDetector({ windowMs: 10000 });
  c.report('LIP_DRIFT', { fired: true, severity: 0.8 }, 1000);
  assert.equal(c.evaluate(2000), null);
  c.report('BLINK_CADENCE', { fired: true, severity: 0.6 }, 5000);
  const m = c.evaluate(5500);
  assert.ok(m);
  assert.deepEqual(m.signals, ['BLINK_CADENCE', 'LIP_DRIFT']);
  assert.ok(m.synergy.length === 1);
});

test('CompoundDetector ignores signals that fell out of the window', () => {
  const c = new CompoundDetector({ windowMs: 10000 });
  c.report('LIP_DRIFT', { fired: true, severity: 0.8 }, 0);
  c.report('BLINK_CADENCE', { fired: true, severity: 0.6 }, 12000);
  assert.equal(c.evaluate(12500), null);
});

test('standing signals raise strength but cannot trigger a marker alone', () => {
  const c = new CompoundDetector();
  c.report('CALLER_AUTH', { fired: true, severity: 0.85 }, 0);
  c.report('GEO_MISMATCH', { fired: true, severity: 0.9 }, 0);
  assert.equal(c.evaluate(1000), null);
  c.report('PROSODY_FLAT', { fired: true, severity: 0.5 }, 1000);
  c.report('BREATH_ABSENT', { fired: true, severity: 0.5 }, 1200);
  const withStanding = c.evaluate(1500);
  const bare = new CompoundDetector();
  bare.report('PROSODY_FLAT', { fired: true, severity: 0.5 }, 1000);
  bare.report('BREATH_ABSENT', { fired: true, severity: 0.5 }, 1200);
  assert.ok(withStanding.strength > bare.evaluate(1500).strength);
});

test('computeTrust falls as independent evidence accumulates', () => {
  const clean = computeTrust([]);
  const one = computeTrust([{ id: 'LIP_DRIFT', severity: 0.8 }]);
  const two = computeTrust([{ id: 'LIP_DRIFT', severity: 0.8 }, { id: 'CALLER_AUTH', severity: 0.85 }]);
  assert.equal(clean, 1);
  assert.ok(one < clean && two < one);
});

test('estimateF0 finds a 200 Hz tone', () => {
  const sr = 48000;
  const buf = new Float32Array(2048);
  for (let i = 0; i < buf.length; i++) buf[i] = 0.3 * Math.sin((2 * Math.PI * 200 * i) / sr) + 0.1 * Math.sin((2 * Math.PI * 400 * i) / sr);
  const { f0, clarity } = estimateF0(buf, sr);
  assert.ok(Math.abs(f0 - 200) < 6, `f0=${f0}`);
  assert.ok(clarity > 0.9);
  assert.ok(rms(buf) > 0.1);
});

test('brickwallDrop detects a cliff at 8 kHz but not a gentle roll-off', () => {
  const sr = 48000;
  const fft = 2048;
  const bins = fft / 2;
  const binHz = sr / fft;
  const cliff = new Float32Array(bins).map((_, i) => (i * binHz < 8000 ? -40 : -115));
  const gentle = new Float32Array(bins).map((_, i) => -40 - (i * binHz) / 400);
  assert.ok(brickwallDrop(cliff, sr, fft).dropDb >= 40);
  assert.ok(brickwallDrop(gentle, sr, fft).dropDb < 10);
});

test('AudioMetrics flags monotone speech without breaths', () => {
  const m = new AudioMetrics();
  let t = 0;
  // 24 s of 1.5 s phrases separated by 0.4 s of digital silence, constant pitch.
  for (let phrase = 0; phrase < 13; phrase++) {
    for (let i = 0; i < 75; i++, t += 20) m.push({ t, dt: 20, rms: 0.1, flatness: 0.05, f0: 120, clarity: 0.9, dropDb: 0 });
    for (let i = 0; i < 20; i++, t += 20) m.push({ t, dt: 20, rms: 0, flatness: 1, f0: 0, clarity: 0, dropDb: 0 });
  }
  const r = Object.fromEntries(m.evaluate().map((x) => [x.id, x]));
  assert.equal(r.PROSODY_FLAT.fired, true);
  assert.equal(r.BREATH_ABSENT.fired, true);
});

test('AudioMetrics accepts natural speech with breaths and pitch movement', () => {
  const m = new AudioMetrics();
  let t = 0;
  for (let phrase = 0; phrase < 13; phrase++) {
    for (let i = 0; i < 75; i++, t += 20) {
      const f0 = 120 * 2 ** (((i % 25) - 12) / 12 * 0.6 + (phrase % 3) * 0.1);
      m.push({ t, dt: 20, rms: 0.1, flatness: 0.05, f0, clarity: 0.9, dropDb: 0 });
    }
    for (let i = 0; i < 20; i++, t += 20) {
      const breath = i >= 4 && i < 14; // 200 ms of noisy low-level breath
      m.push({ t, dt: 20, rms: breath ? 0.006 : 0.0005, flatness: breath ? 0.6 : 0.2, f0: 0, clarity: 0, dropDb: 0 });
    }
  }
  const r = Object.fromEntries(m.evaluate().map((x) => [x.id, x]));
  assert.equal(r.PROSODY_FLAT.fired, false);
  assert.equal(r.BREATH_ABSENT.fired, false);
});

// Build a synthetic 468-point face with controllable eye openness.
function face(eyeOpen = 0.3, jawOpen = 0.05, jitter = 0) {
  const lm = Array.from({ length: 468 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const set = (i, x, y) => { lm[i] = { x: x + jitter * (Math.random() - 0.5), y: y + jitter * (Math.random() - 0.5), z: 0 }; };
  set(10, 0.5, 0.2); set(152, 0.5, 0.8); set(1, 0.5, 0.5);
  set(13, 0.5, 0.65); set(14, 0.5, 0.65 + jawOpen);
  set(61, 0.45, 0.67); set(291, 0.55, 0.67);
  const eye = (ids, cx) => {
    const [p1, p2, p3, p4, p5, p6] = ids;
    set(p1, cx - 0.03, 0.4); set(p4, cx + 0.03, 0.4);
    set(p2, cx - 0.01, 0.4 - eyeOpen * 0.03); set(p3, cx + 0.01, 0.4 - eyeOpen * 0.03);
    set(p6, cx - 0.01, 0.4 + eyeOpen * 0.03); set(p5, cx + 0.01, 0.4 + eyeOpen * 0.03);
  };
  eye([33, 160, 158, 133, 153, 144], 0.4);
  eye([362, 385, 387, 263, 373, 380], 0.6);
  return lm;
}

function runVideo(blinkTimes, seconds = 40) {
  const vm = new VideoMetrics();
  for (let t = 0; t < seconds * 1000; t += 33) {
    const blinking = blinkTimes.some((b) => t >= b && t < b + 150);
    vm.push(t, face(blinking ? 0.02 : 0.3, 0.05 + 0.03 * Math.sin(t / 300), 0.0004), null);
  }
  return Object.fromEntries(vm.evaluate().map((r) => [r.id, r]));
}

test('VideoMetrics: natural blinking passes, no blinking fires', () => {
  const irregular = [2500, 6100, 9000, 14500, 17200, 22800, 25000, 31000, 34200, 38000];
  assert.equal(runVideo(irregular).BLINK_CADENCE.fired, false);
  assert.equal(runVideo([]).BLINK_CADENCE.fired, true);
});

test('VideoMetrics: metronomic blinking fires', () => {
  const regular = Array.from({ length: 13 }, (_, i) => 2000 + i * 3000);
  const r = runVideo(regular);
  assert.equal(r.BLINK_CADENCE.fired, true);
  assert.ok(r.BLINK_CADENCE.detail.includes('metronomic'));
});

test('gestures on synthetic hands', () => {
  const hand = (over = {}) => {
    const h = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.6, z: 0 }));
    h[0] = { x: 0.5, y: 0.8, z: 0 };
    for (const [pip, tip] of [[6, 8], [10, 12], [14, 16], [18, 20]]) { h[pip] = { x: 0.5, y: 0.7, z: 0 }; h[tip] = { x: 0.5, y: 0.75, z: 0 }; } // curled
    h[2] = { x: 0.45, y: 0.72, z: 0 }; h[3] = { x: 0.44, y: 0.65, z: 0 }; h[4] = { x: 0.44, y: 0.55, z: 0 };
    return Object.assign(h, over);
  };
  assert.equal(isThumbsUp(hand()), true);
  assert.equal(isPeace(hand()), false);
  const peace = hand({ 8: { x: 0.5, y: 0.3, z: 0 }, 12: { x: 0.52, y: 0.3, z: 0 } });
  assert.equal(isPeace(peace), true);
  const f = face();
  const near = hand({ 8: { x: 0.5, y: 0.51, z: 0 } });
  assert.equal(isTouchingNose(near, f, 1), true);
  assert.equal(isTouchingNose(hand(), f, 1), false);
});

// Head turning, as it looks to the pose vector: the vector is centred and divided by eye distance, so
// translation cancels and what remains is the geometric foreshortening of a yaw/pitch change.
let noiseSeed = 99;
const noise = () => ((noiseSeed = (noiseSeed * 48271) % 2147483647) / 2147483647 - 0.5);

function turningFace(yaw, pitch, jitter) {
  const c = Math.cos(yaw);
  return face(0.3, 0.05, 0).map((p) => ({
    x: 0.5 + (p.x - 0.5) * c + jitter * noise(),
    y: p.y + pitch + jitter * noise(),
    z: 0,
  }));
}

// `dropEvery` simulates MediaPipe returning nothing because the video frame has not advanced.
function runMotion({ jitter, dropEvery = 0, seconds = 8 }) {
  const vm = new VideoMetrics();
  let frame = 0;
  for (let t = 0; t < seconds * 1000; t += 33) {
    frame++;
    if (dropEvery && frame % dropEvery === 0) continue;
    const yaw = 0.6 * Math.sin((2 * Math.PI * t) / 1000);
    vm.push(t, turningFace(yaw, 0.02 * Math.sin((2 * Math.PI * t) / 1700), jitter), null);
  }
  return vm;
}

test('MOTION_SMOOTHNESS: a moving head with dropped frames is not flagged', () => {
  const vm = runMotion({ jitter: 0.0006, dropEvery: 5 });
  const r = vm.evaluate().find((x) => x.id === 'MOTION_SMOOTHNESS');
  assert.ok(r, 'signal is evaluated');
  assert.equal(r.fired, false, r.detail);
  // The gaps are real and would wreck an unfiltered second difference, so the filter must be doing the work.
  assert.ok(vm.jitterStats().dropped > 10, 'frames were actually dropped');
});

test('MOTION_SMOOTHNESS: per-frame landmark flicker is flagged', () => {
  const r = runMotion({ jitter: 0.02 }).evaluate().find((x) => x.id === 'MOTION_SMOOTHNESS');
  assert.equal(r.fired, true, r.detail);
  assert.ok(r.detail.includes('flickers'));
});

test('MOTION_SMOOTHNESS: still frames do not fire on noise ratio alone', () => {
  const vm = new VideoMetrics();
  for (let t = 0; t < 8000; t += 33) vm.push(t, turningFace(0, 0, 0.0006), null);
  const r = vm.evaluate().find((x) => x.id === 'MOTION_SMOOTHNESS');
  assert.equal(r.fired, false, r.detail);
});

test('probes report readiness before a signal can be judged', () => {
  const vm = runMotion({ jitter: 0.0006, seconds: 3 });
  const p = Object.fromEntries(vm.probe().map((x) => [x.key, x]));
  assert.equal(p.blink.progress < 1, true);
  assert.equal(p.jitter.progress, 1);
  assert.ok(p.jitter.text.includes('ratio'));
  assert.ok(p.face.text.includes('fps'));
});

// --- cumulative trust ---

function runLedger(ledger, firing, seconds, startT = 0, stepMs = 1000) {
  let t = startT;
  for (let i = 0; i < seconds; i++) {
    ledger.observe(firing, t);
    t += stepMs;
  }
  return t;
}

test('TrustLedger keeps the score down after a signal stops firing', () => {
  const l = new TrustLedger();
  const t = runLedger(l, [{ id: 'LIP_DRIFT', severity: 0.8 }], 20);
  const during = l.trust();
  assert.ok(during < 0.5, `trust during evidence was ${during}`);

  // Nothing firing for 10 s: the score must not snap back to clean.
  l.observe([], t + 10000);
  const after = l.trust();
  assert.ok(after < 0.65, `trust right after evidence was ${after}`);
  assert.ok(after > during, 'it should recover somewhat');
});

test('TrustLedger decays a single blip to near-clean over several half-lives', () => {
  const l = new TrustLedger({ halfLifeMs: 1000 });
  l.observe([{ id: 'LIP_DRIFT', severity: 0.8 }], 0);
  const spike = l.trust();
  assert.ok(spike < 1);
  l.observe([], 12000);
  assert.ok(l.trust() > 0.98, `a lone blip should fade, got ${l.trust()}`);
});

test('TrustLedger compounds sustained evidence beyond a single observation', () => {
  const brief = new TrustLedger();
  brief.observe([{ id: 'BLINK_CADENCE', severity: 0.6 }], 0);
  const sustained = new TrustLedger();
  runLedger(sustained, [{ id: 'BLINK_CADENCE', severity: 0.6 }], 15);
  assert.ok(sustained.trust() < brief.trust() - 0.1, 'sustained evidence must weigh more than one tick');
});

test('TrustLedger applies standing signals at full weight without accumulating them', () => {
  const l = new TrustLedger();
  l.observe([], 0);
  const withStanding = l.trust([{ id: 'CALLER_AUTH', severity: 1 }]);
  assert.ok(Math.abs(withStanding - (1 - SIGNALS.CALLER_AUTH.weight)) < 1e-9);
  assert.equal(l.trust(), 1, 'standing signals are not added to the ledger');
});

test('TrustLedger summary ranks the strongest evidence first', () => {
  const l = new TrustLedger();
  runLedger(l, [{ id: 'LIP_DRIFT', severity: 0.9 }, { id: 'EXPRESSION_FLAT', severity: 0.3 }], 10);
  const s = l.summary();
  assert.equal(s[0].id, 'LIP_DRIFT');
  assert.ok(s[0].contribution > s[1].contribution);
  assert.ok(s[0].seconds >= 9);
});

test('TrustLedger resets', () => {
  const l = new TrustLedger();
  runLedger(l, [{ id: 'LIP_DRIFT', severity: 0.9 }], 5);
  l.reset();
  assert.equal(l.trust(), 1);
  assert.deepEqual(l.summary(), []);
});

// --- multi-face tracking ---

const boxFace = (x, y, s = 0.2) => ({ box: { x, y, w: s, h: s } });

test('FaceTracker keeps an identity when the detector reorders faces', () => {
  const tr = new FaceTracker();
  const a = tr.update([boxFace(0.1, 0.1, 0.3), boxFace(0.6, 0.1, 0.2)], 0);
  assert.deepEqual(a.map((f) => f.id), [1, 2]);
  assert.deepEqual(a.map((f) => f.primary), [true, false]);

  // Same two people, emitted in the other order: ids must follow the face, not the position in the list.
  const b = tr.update([boxFace(0.6, 0.1, 0.2), boxFace(0.1, 0.1, 0.3)], 100);
  assert.deepEqual(b.map((f) => f.id), [2, 1]);
  assert.equal(b[0].primary, true, 'primary is whichever the caller listed first');
});

test('FaceTracker follows a face that moves', () => {
  const tr = new FaceTracker();
  tr.update([boxFace(0.1, 0.1)], 0);
  const moved = tr.update([boxFace(0.16, 0.14)], 100);
  assert.equal(moved[0].id, 1);
});

test('FaceTracker issues a new identity after a long absence', () => {
  const tr = new FaceTracker();
  tr.update([boxFace(0.1, 0.1)], 0);
  assert.equal(tr.update([boxFace(0.1, 0.1)], 5000)[0].id, 2);
});

test('FaceTracker gives separate people separate identities', () => {
  const tr = new FaceTracker();
  const r = tr.update([boxFace(0.05, 0.05), boxFace(0.7, 0.7)], 0);
  assert.equal(new Set(r.map((f) => f.id)).size, 2);
});
