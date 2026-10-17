"""Tests for the ported heuristics.

The expected values here are the ones the JavaScript suite asserts (see test/signals.test.js and
test/analysis.test.js), so if the two implementations ever drift apart these fail.

Run with: npm test  (or: venv/Scripts/python -m unittest discover -s tests)
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from analysis import AnalysisSession  # noqa: E402
from analysis.features import brickwall_drop, estimate_f0, rms, spectrum_db  # noqa: E402
from analysis.metrics import (  # noqa: E402
    BLEND_KEYS, LM_IDS, AudioMetrics, CrossModal, VideoMetrics, cross_correlate, unpack_face,
)

SR = 32000


def envelope(seconds, seed=3):
    """Speech-like bursts of varying length, from the same generator the JS tests use."""
    s = seed
    def rnd():
        nonlocal s
        s = (s * 16807) % 2147483647
        return s / 2147483647

    bursts, t = [], 0.2
    while t < seconds + 2:
        on = 0.15 + rnd() * 0.25
        bursts.append((t, on, 0.5 + rnd() * 0.5))
        t += on + 0.1 + rnd() * 0.25

    def at(tt):
        for t0, on, amp in bursts:
            if t0 <= tt < t0 + on:
                return amp * math.sin(math.pi * (tt - t0) / on)
        return 0.0
    return at


def voice(t, env):
    return env(t) * 0.3 * (math.sin(2 * math.pi * 140 * t)
                           + 0.5 * math.sin(2 * math.pi * 280 * t)
                           + 0.3 * math.sin(2 * math.pi * 420 * t))


def packed_face(mouth=0.05, blend=None):
    p = []
    for i in LM_IDS:
        x, y = 0.5 + i * 1e-3, 0.5 + i * 1e-3
        if i == 10:
            x, y = 0.5, 0.2
        if i == 152:
            x, y = 0.5, 0.8
        if i == 13:
            x, y = 0.5, 0.5
        if i == 14:
            x, y = 0.5, 0.5 + mouth * 0.6
        p += [x, y]
    if blend:
        p += [blend.get(k, 0) for k in BLEND_KEYS]
    return p


def drive(session, seconds, lag_sec=0.0, chunk=640, clock=5000):
    env = envelope(seconds)
    next_face = 0
    for i in range(0, int(seconds * SR), chunk):
        x = np.array([voice((i + k) / SR - lag_sec, env) for k in range(chunk)], dtype=np.float32)
        end_t = clock + ((i + chunk) / SR) * 1000
        session.push_audio(x, end_t)
        while next_face <= end_t:
            session.push_face(next_face, packed_face(0.02 + 0.1 * env((next_face - clock) / 1000)))
            next_face += 40
    return session


class TestFeatures(unittest.TestCase):
    def test_spectrum_puts_a_tone_in_the_right_bin(self):
        n = 2048
        x = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(n) / SR)
        peak = int(np.argmax(spectrum_db(x)))
        self.assertLessEqual(abs(peak - round(1000 * n / SR)), 1)

    def test_estimate_f0_recovers_a_known_pitch(self):
        n = 2048
        t = np.arange(n) / SR
        x = (np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        f0, clarity = estimate_f0(x, SR)
        self.assertLess(abs(f0 - 150), 4, f"got {f0}")
        self.assertGreater(clarity, 0.8)

    def test_estimate_f0_is_silent_on_noise_free_silence(self):
        self.assertEqual(estimate_f0(np.zeros(2048, dtype=np.float32), SR), (0.0, 0.0))

    def test_rms(self):
        self.assertAlmostEqual(rms(np.array([1.0, -1.0, 1.0, -1.0])), 1.0)
        self.assertEqual(rms(np.zeros(10)), 0.0)

    def test_brickwall_drop_finds_a_cliff_but_not_a_rolloff(self):
        n = 2048
        hz = SR / n
        bins = np.arange(n // 2) * hz
        cliff = np.where(bins < 8000, -20.0, -100.0)
        self.assertGreater(brickwall_drop(cliff, SR, n)[0], 40)
        self.assertAlmostEqual(brickwall_drop(cliff, SR, n)[1], 8000, delta=600)
        rolloff = -20 - (bins / 16000) * 40
        self.assertLess(brickwall_drop(rolloff, SR, n)[0], 40)

    def test_cross_correlate_recovers_a_known_lag(self):
        env = envelope(10)
        x = [env(i / 25) for i in range(250)]
        shift = 6
        y = [x[i - shift] if i - shift >= 0 else 0.0 for i in range(250)]
        lag, corr = cross_correlate(x, y, 12)
        self.assertEqual(lag, shift)
        self.assertGreater(corr, 0.9)


class TestWire(unittest.TestCase):
    def test_round_trips_the_landmarks_the_metrics_read(self):
        blend = {k: i / 20 for i, k in enumerate(BLEND_KEYS)}
        lm, got = unpack_face(packed_face(0.1, blend))
        self.assertAlmostEqual(lm[14][1], 0.56, places=4)
        self.assertEqual(got["jawOpen"], 0)
        self.assertIsNone(unpack_face(packed_face(0.1))[1])

    def test_rejects_malformed_input(self):
        self.assertEqual(unpack_face([1, 2, 3]), (None, None))
        self.assertEqual(unpack_face(None), (None, None))
        self.assertEqual(unpack_face([float("nan")] * 40), (None, None))


class TestAudioMetrics(unittest.TestCase):
    def _run(self, *, monotone):
        m = AudioMetrics()
        env = envelope(40)
        t = 0.0
        while t < 40000:
            secs = t / 1000
            level = 0.3 * env(secs)
            speaking = level >= 0.02
            f0 = 150 if monotone else 150 * (1 + 0.12 * math.sin(secs * 1.7))
            # Natural speech: audible breath noise in the pauses before phrases.
            flat = 0.1 if speaking else 0.4
            lvl = level if speaking else (0.0001 if monotone else 0.005)
            m.push(t, 20, lvl, flat, f0 if speaking else 0, 0.9 if speaking else 0, 0)
            t += 20
        return {r["id"]: r for r in m.evaluate()}

    def test_flags_monotone_speech_without_breaths(self):
        r = self._run(monotone=True)
        self.assertTrue(r["PROSODY_FLAT"]["fired"], r["PROSODY_FLAT"]["detail"])
        self.assertTrue(r["BREATH_ABSENT"]["fired"], r["BREATH_ABSENT"]["detail"])

    def test_accepts_speech_with_pitch_movement_and_breaths(self):
        r = self._run(monotone=False)
        self.assertFalse(r["PROSODY_FLAT"]["fired"], r["PROSODY_FLAT"]["detail"])
        self.assertFalse(r["BREATH_ABSENT"]["fired"], r["BREATH_ABSENT"]["detail"])


def video_face(eye_open=0.3, jaw=0.05, yaw=0.0, jitter=0.0, rng=None):
    """A face as the pose vector sees it: centred and scaled, so only geometry changes survive."""
    lm = {}
    base = {10: (0.5, 0.2), 152: (0.5, 0.8), 1: (0.5, 0.5), 13: (0.5, 0.65), 14: (0.5, 0.65 + jaw),
            61: (0.45, 0.67), 291: (0.55, 0.67)}
    for ids, cx in (([33, 160, 158, 133, 153, 144], 0.4), ([362, 385, 387, 263, 373, 380], 0.6)):
        p1, p2, p3, p4, p5, p6 = ids
        base[p1] = (cx - 0.03, 0.4)
        base[p4] = (cx + 0.03, 0.4)
        base[p2] = (cx - 0.01, 0.4 - eye_open * 0.03)
        base[p3] = (cx + 0.01, 0.4 - eye_open * 0.03)
        base[p6] = (cx - 0.01, 0.4 + eye_open * 0.03)
        base[p5] = (cx + 0.01, 0.4 + eye_open * 0.03)
    c = math.cos(yaw)
    for i in LM_IDS:
        x, y = base.get(i, (0.5, 0.5))
        x = 0.5 + (x - 0.5) * c
        if jitter and rng is not None:
            x += jitter * (rng.random() - 0.5)
            y += jitter * (rng.random() - 0.5)
        lm[i] = (x, y)
    return lm


class TestVideoMetrics(unittest.TestCase):
    def _blinks(self, blink_times, seconds=40):
        vm = VideoMetrics()
        for t in range(0, seconds * 1000, 33):
            blinking = any(b <= t < b + 150 for b in blink_times)
            vm.push(t, video_face(0.02 if blinking else 0.3, 0.05 + 0.03 * math.sin(t / 300)), None)
        return {r["id"]: r for r in vm.evaluate()}

    def test_natural_blinking_passes_and_no_blinking_fires(self):
        irregular = [2500, 6100, 9000, 14500, 17200, 22800, 25000, 31000, 34200, 38000]
        self.assertFalse(self._blinks(irregular)["BLINK_CADENCE"]["fired"])
        self.assertTrue(self._blinks([])["BLINK_CADENCE"]["fired"])

    def test_metronomic_blinking_fires(self):
        r = self._blinks([2000 + i * 3000 for i in range(13)])
        self.assertTrue(r["BLINK_CADENCE"]["fired"])
        self.assertIn("metronomic", r["BLINK_CADENCE"]["detail"])

    def _motion(self, jitter, drop_every=0, seconds=8):
        vm = VideoMetrics()
        rng = np.random.RandomState(99)
        frame = 0
        for t in range(0, seconds * 1000, 33):
            frame += 1
            if drop_every and frame % drop_every == 0:
                continue
            yaw = 0.6 * math.sin(2 * math.pi * t / 1000)
            vm.push(t, video_face(yaw=yaw, jitter=jitter, rng=rng), None)
        return vm

    def test_moving_head_with_dropped_frames_is_not_flagged(self):
        vm = self._motion(0.0006, drop_every=5)
        r = {x["id"]: x for x in vm.evaluate()}["MOTION_SMOOTHNESS"]
        self.assertFalse(r["fired"], r["detail"])
        self.assertGreater(vm.jitter_stats()["dropped"], 10, "frames were actually dropped")

    def test_per_frame_flicker_is_flagged(self):
        r = {x["id"]: x for x in self._motion(0.02).evaluate()}["MOTION_SMOOTHNESS"]
        self.assertTrue(r["fired"], r["detail"])
        self.assertIn("flickers", r["detail"])

    def test_still_frames_do_not_fire_on_noise_ratio_alone(self):
        vm = VideoMetrics()
        rng = np.random.RandomState(7)
        for t in range(0, 8000, 33):
            vm.push(t, video_face(jitter=0.0006, rng=rng), None)
        r = {x["id"]: x for x in vm.evaluate()}["MOTION_SMOOTHNESS"]
        self.assertFalse(r["fired"], r["detail"])


class TestSession(unittest.TestCase):
    def test_recovers_a_300ms_audio_lag(self):
        s = drive(AnalysisSession(sample_rate=SR, model=None), 12, lag_sec=0.3)
        # A constant offset is a capture-pipeline property, so it is measured but not flagged; what the
        # session has to get right here is the number.
        m = s.cross.measure(s.now)
        self.assertIsNotNone(m)
        self.assertLessEqual(abs(m["lag_ms"] - 300), 80, f"measured {m['lag_ms']} ms")
        lip = next(r for r in s.evaluate() if r["id"] == "LIP_DRIFT")
        self.assertFalse(lip["fired"], lip["detail"])

    def test_aligned_audio_is_in_sync(self):
        s = drive(AnalysisSession(sample_rate=SR, model=None), 12, lag_sec=0.0)
        m = s.cross.measure(s.now)
        self.assertLessEqual(abs(m["lag_ms"]), 80, f"measured {m['lag_ms']} ms")
        lip = next(r for r in s.evaluate() if r["id"] == "LIP_DRIFT")
        self.assertFalse(lip["fired"], lip["detail"])

    def test_probes_report_readiness(self):
        s = drive(AnalysisSession(sample_rate=SR, model=None), 6)
        probes = s.probes()
        self.assertEqual(set(probes), {"video", "voice", "crossmodal"})
        blink = next(p for p in probes["video"] if p["key"] == "blink")
        self.assertLess(blink["progress"], 1)
        self.assertFalse(any(p["key"] == "model" for p in probes["voice"]), "no model configured")

    def test_without_a_model_no_voice_model_signal(self):
        s = drive(AnalysisSession(sample_rate=SR, model=None), 6)
        self.assertEqual([r for r in s.evaluate() if r["id"] == "VOICE_MODEL"], [])

    def test_reset_clears_everything(self):
        s = drive(AnalysisSession(sample_rate=SR, model=None), 6)
        self.assertGreater(s.frames, 0)
        s.reset()
        self.assertEqual(s.frames, 0)
        self.assertEqual(s.evaluate(), [])


class TestVoiceModelGating(unittest.TestCase):
    class StubModel:
        ready = True
        busy = False

        def __init__(self, score):
            self.score = score
            self.calls = 0

        def submit(self, pcm, callback):
            self.calls += 1
            callback(self.score)
            return True

    def test_fires_only_after_several_agreeing_windows(self):
        stub = self.StubModel(0.97)
        s = drive(AnalysisSession(sample_rate=SR, model=stub), 14)
        self.assertGreaterEqual(stub.calls, 3, f"{stub.calls} windows scored")
        r = next(x for x in s.evaluate() if x["id"] == "VOICE_MODEL")
        self.assertTrue(r["fired"], r["detail"])
        self.assertIn("97%", r["detail"])

    def test_a_confidently_real_voice_does_not_fire(self):
        s = drive(AnalysisSession(sample_rate=SR, model=self.StubModel(0.04)), 14)
        r = next(x for x in s.evaluate() if x["id"] == "VOICE_MODEL")
        self.assertFalse(r["fired"], r["detail"])

    def test_silence_is_never_scored(self):
        stub = self.StubModel(0.99)
        s = AnalysisSession(sample_rate=SR, model=stub)
        for i in range(0, 12 * SR, 640):
            s.push_audio(np.zeros(640, dtype=np.float32), 5000 + ((i + 640) / SR) * 1000)
        self.assertEqual(stub.calls, 0)

    def test_unvoiced_noise_is_never_scored(self):
        stub = self.StubModel(0.99)
        s = AnalysisSession(sample_rate=SR, model=stub)
        rng = np.random.RandomState(1)
        for i in range(0, 12 * SR, 640):
            noise = (rng.random_sample(640).astype(np.float32) - 0.5) * 0.6
            s.push_audio(noise, 5000 + ((i + 640) / SR) * 1000)
        self.assertEqual(stub.calls, 0, "noise must not reach the model")


if __name__ == "__main__":
    unittest.main()


def run_cross_modal(seconds, shift_at, hz=50):
    """Feeds a CrossModal in real-time order, evaluating once a second as the server does.

    `shift_at(t)` gives the audio delay in samples at that moment, so a capture pipeline's constant offset
    and a mid-call change can both be simulated.
    """
    env = envelope(seconds + 4)
    cm = CrossModal()
    next_eval = 1000
    last = []
    for i in range(int(seconds * hz)):
        t = i * (1000 / hz)
        cm.push_mouth(t, 0.02 + 0.1 * env(i / hz))
        cm.push_audio(t, 0.01 + 0.2 * env((i - shift_at(t)) / hz))
        if t >= next_eval:
            r = cm.evaluate(t)
            if r:
                last = r
            next_eval += 1000
    return cm, {x["id"]: x for x in last}


class TestCrossModalBaseline(unittest.TestCase):
    def test_constant_device_offset_is_the_baseline_not_drift(self):
        # 300 ms of constant audio delay is ordinary webcam latency and must not fire.
        cm, by_id = run_cross_modal(24, lambda t: 15)
        self.assertFalse(by_id["LIP_DRIFT"]["fired"], by_id["LIP_DRIFT"]["detail"])
        self.assertIsNotNone(cm.baseline)
        self.assertLess(abs(cm.baseline - 300), 60, f"baseline was {cm.baseline}")
        self.assertIn("baseline", by_id["LIP_DRIFT"]["detail"])

    def test_does_not_fire_while_learning_the_baseline(self):
        cm, by_id = run_cross_modal(12, lambda t: 15)
        self.assertIsNone(cm.baseline)
        self.assertFalse(by_id["LIP_DRIFT"]["fired"])
        self.assertIn("Learning", by_id["LIP_DRIFT"]["detail"])

    def test_flags_sync_that_shifts_away_from_the_baseline(self):
        _, by_id = run_cross_modal(40, lambda t: 0 if t < 22000 else 20)
        self.assertTrue(by_id["LIP_DRIFT"]["fired"], by_id["LIP_DRIFT"]["detail"])
        self.assertRegex(by_id["LIP_DRIFT"]["detail"], "shifted|wandered")

    def test_gross_offset_fires_before_a_baseline_exists(self):
        cm, by_id = run_cross_modal(12, lambda t: 36)  # 720 ms, beyond any capture pipeline
        self.assertIsNone(cm.baseline, "still calibrating")
        self.assertTrue(by_id["LIP_DRIFT"]["fired"], by_id["LIP_DRIFT"]["detail"])
        self.assertIn("beyond any capture delay", by_id["LIP_DRIFT"]["detail"])

    def test_normally_correlated_call_stays_quiet(self):
        _, by_id = run_cross_modal(24, lambda t: 0)
        self.assertFalse(by_id["LIP_DRIFT"]["fired"], by_id["LIP_DRIFT"]["detail"])
        self.assertFalse(by_id["LIP_DECOUPLED"]["fired"], by_id["LIP_DECOUPLED"]["detail"])

    def test_decoupled_needs_sustained_loss_of_correspondence(self):
        cm = CrossModal()
        cm.measure = lambda now: {"lag_ms": 0, "peak": 0.05}
        fired = [next(x for x in cm.evaluate(1000 * i) if x["id"] == "LIP_DECOUPLED")["fired"]
                 for i in range(4)]
        self.assertEqual(fired, [False, False, True, True])

        # One good reading clears the run, so an isolated dip cannot accumulate into an alert.
        cm.measure = lambda now: {"lag_ms": 0, "peak": 0.6}
        self.assertFalse(next(x for x in cm.evaluate(5000) if x["id"] == "LIP_DECOUPLED")["fired"])
        self.assertEqual(cm.low_corr_run, 0)
