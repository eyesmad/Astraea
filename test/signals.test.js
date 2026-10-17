import test from 'node:test';
import assert from 'node:assert/strict';
import { crossCorrelate, resample, zscore } from '../public/js/crosscorr.js';
import { CrossModal } from '../public/js/crossmodal.js';
import { CompoundDetector, computeTrust } from '../public/js/compound.js';
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
