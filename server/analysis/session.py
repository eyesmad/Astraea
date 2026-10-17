"""One analysis session per capture: turns streamed PCM and face frames into signals and live read-outs."""
import os

import numpy as np

from .features import brickwall_drop, decimate2, estimate_f0, rms, spectral_flatness, spectrum_db
from .metrics import AudioMetrics, CrossModal, VideoMetrics, mouth_aperture, unpack_face
from .voice_model import voice_model

SAMPLE_RATES = {16000, 32000, 44100, 48000}

# The model reads a 4 s window. Scoring is slower than real time on CPU, so windows are scored at an interval
# rather than continuously, and a run of agreeing scores is required before the signal fires: one window is
# noisy, and this signal is weighted heavily enough that a lone spike would mislead.
MODEL_WINDOW_SEC = 4
MODEL_INTERVAL_MS = 3000
MODEL_MIN_SCORES = 3
MODEL_THRESHOLD = 0.85
MODEL_SPEECH_RMS = 0.02
# The model is confident on anything that is not speech: a sine tone and white noise both score over 99%
# synthetic. Scoring only windows with voiced audio keeps hold music and line noise out of it.
MODEL_MIN_VOICED = 25


class AnalysisSession:
    def __init__(self, sample_rate=32000, fft_size=None, hop_ms=20, model=None):
        self.sr = sample_rate if sample_rate in SAMPLE_RATES else 32000
        self.fft_size = fft_size or (2048 if self.sr > 24000 else 1024)
        self.hop = int(round((self.sr * hop_ms) / 1000))
        self.hop_ms = (self.hop / self.sr) * 1000
        self.model = voice_model() if model is None else model
        self.audio = AudioMetrics()
        self.video = VideoMetrics()
        self.cross = CrossModal()
        self.reset()

    def reset(self):
        self.audio.reset()
        self.video.reset()
        self.cross.reset()
        self.ring = np.zeros(self.fft_size, dtype=np.float32)
        self.filled = 0
        self.since_hop = 0
        self.total = 0
        self.offset = None
        self.now = 0.0
        self.tick = 0
        self.f0 = (0.0, 0.0)
        self.drop = (0.0, 0.0)
        self.last_frame_t = 0.0
        self.model_ring = np.zeros(self.sr * MODEL_WINDOW_SEC, dtype=np.float32)
        self.model_filled = 0
        self.model_scores = []
        self.last_score_t = 0.0
        self.faces = 0  # faces visible in the most recent frame

    @property
    def frames(self):
        return self.video.frames

    def push_audio(self, samples, end_t):
        """samples: float32 in [-1, 1]; end_t: client clock (ms) when the last sample was captured."""
        n = len(samples)
        if n == 0:
            return
        end_idx = self.total + n
        off = end_t - (end_idx / self.sr) * 1000
        # Network jitter only ever adds delay, so track the lower envelope of the offset estimates.
        self.offset = off if self.offset is None else min(off, self.offset + 0.02)

        size = self.fft_size
        model_size = len(self.model_ring)
        for i in range(n):
            pos = self.total % size
            self.ring[pos] = samples[i]
            self.model_ring[self.total % model_size] = samples[i]
            if self.model_filled < model_size:
                self.model_filled += 1
            self.total += 1
            if self.filled < size:
                self.filled += 1
            self.since_hop += 1
            if self.since_hop >= self.hop:
                self.since_hop = 0
                if self.filled == size:
                    self._frame(self.offset + (self.total / self.sr) * 1000)

    def _frame(self, t):
        size = self.fft_size
        start = self.total % size
        frame = np.concatenate((self.ring[start:], self.ring[:start])) if start else self.ring.copy()

        level = rms(frame)
        self.cross.push_audio(t, level)
        self.now = max(self.now, t)

        freq = spectrum_db(frame)
        hz = self.sr / size
        flat = spectral_flatness(freq, int(300 / hz), int(4000 / hz))
        # Pitch and spectral-cliff analysis are heavier, so run them on every 2nd frame of voiced audio.
        if level >= 0.02 and self.tick % 2 == 0:
            self.f0 = estimate_f0(frame, self.sr)
            self.drop = brickwall_drop(freq, self.sr, size)
        if level >= 0.02:
            self.tick += 1

        dt = (t - self.last_frame_t) if self.last_frame_t else self.hop_ms
        self.last_frame_t = t
        self.audio.push(t, dt, level, flat, self.f0[0], self.f0[1], self.drop[0])
        self._maybe_score(t, level)

    def _maybe_score(self, t, level):
        model = self.model
        if model is None or model.busy:
            return
        if self.model_filled < len(self.model_ring):
            return
        if level < MODEL_SPEECH_RMS:
            return
        if self.last_score_t and t - self.last_score_t < MODEL_INTERVAL_MS:
            return
        window_start = t - MODEL_WINDOW_SEC * 1000
        if sum(1 for p in self.audio.f0 if p["t"] >= window_start) < MODEL_MIN_VOICED:
            return
        self.last_score_t = t

        size = len(self.model_ring)
        start = self.total % size
        pcm = np.concatenate((self.model_ring[start:], self.model_ring[:start])) if start else self.model_ring.copy()
        rate = self.sr
        while rate > 16000:
            pcm = decimate2(pcm)
            rate //= 2
        model.submit(pcm, lambda score, when=t: self._record_score(when, score))

    def _record_score(self, t, score):
        if score is None:
            return
        self.model_scores.append({"t": t, "score": score})
        if len(self.model_scores) > 8:
            self.model_scores.pop(0)

    def model_verdict(self):
        """Median of recent scores: one odd window should not move a signal this heavily weighted."""
        if len(self.model_scores) < MODEL_MIN_SCORES:
            return None
        v = sorted(s["score"] for s in self.model_scores)
        return {"median": v[len(v) // 2], "n": len(v)}

    def evaluate_model(self):
        v = self.model_verdict()
        if not v:
            return []
        fired = v["median"] >= MODEL_THRESHOLD
        return [{
            "id": "VOICE_MODEL",
            "fired": fired,
            "severity": min(1.0, 0.5 + (v["median"] - MODEL_THRESHOLD) * 2) if fired else 0,
            "detail": f"Voice model scores this {v['median'] * 100:.0f}% synthetic over {v['n']} windows"
                      + ("." if fired else ", below the alert threshold."),
        }]

    def push_face(self, t, packed):
        lm, blend = unpack_face(packed)
        if lm is None:
            return False
        self.video.push(t, lm, blend)
        self.cross.push_mouth(t, mouth_aperture(lm))
        self.now = max(self.now, t)
        return True

    def evaluate(self):
        return [*self.video.evaluate(), *self.audio.evaluate(),
                *self.cross.evaluate(self.now), *self.evaluate_model()]

    def probes(self):
        voice = self.audio.probe()
        if self.model is not None:
            v = self.model_verdict()
            voice.append({
                "key": "model", "label": "Voice model",
                "progress": min(1.0, len(self.model_scores) / MODEL_MIN_SCORES),
                "text": f"{v['median'] * 100:.0f}% synthetic over {v['n']} windows" if v
                        else ("warming up" if self.model.ready else "loading model"),
                "bar": min(1.0, max(0.0, v["median"])) if v else 0,
            })
        return {"video": self.video.probe(), "voice": voice, "crossmodal": self.cross.probe(self.now)}
