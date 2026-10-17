"""Voice, face and cross-modal heuristics.

Ported from public/js/audioMetrics.js, videoMetrics.js and crossmodal.js with the same thresholds, so the
behaviour the JS tests pin is the behaviour the server runs. Thresholds remain untuned proof-of-concept
values; calibrate them against recorded calls rather than trusting the constants.
"""
import math

import numpy as np

from .features import clamp, median, percentile, robust_std, std

SPEECH_RMS = 0.02
DIGITAL_SILENCE = 2e-4

# Landmark indices, mirroring IDX in public/js/videoMetrics.js.
IDX = {
    "rightEye": [33, 160, 158, 133, 153, 144],
    "leftEye": [362, 385, 387, 263, 373, 380],
    "noseTip": 1,
    "forehead": 10,
    "chin": 152,
    "lipTop": 13,
    "lipBottom": 14,
    "stable": [1, 152, 33, 263, 61, 291, 10],
}

BLEND_KEYS = [
    "jawOpen", "mouthSmileLeft", "mouthSmileRight", "browDownLeft", "browDownRight",
    "browInnerUp", "eyeSquintLeft", "eyeSquintRight", "mouthFrownLeft", "mouthFrownRight",
    "cheekPuff", "mouthPucker",
]


def _lm_ids():
    """Same order as LM_IDS in public/js/wire.js: a de-duplicated concatenation that preserves first use."""
    order = (IDX["rightEye"] + IDX["leftEye"] + [IDX["noseTip"], IDX["forehead"], IDX["chin"],
             IDX["lipTop"], IDX["lipBottom"]] + IDX["stable"])
    seen, out = set(), []
    for i in order:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


LM_IDS = _lm_ids()

# MOTION_SMOOTHNESS thresholds; see the note in public/js/videoMetrics.js. For white landmark noise the
# ratio rms(2nd difference)/rms(1st difference) tends to sqrt(3) = 1.73, while smooth head motion sits well
# below it because velocity dominates acceleration over one frame.
JITTER_FLOOR = 0.025
NOISE_RATIO = 1.55
MIN_MOTION = 0.004


def unpack_face(p):
    """Turns the packed wire array back into (landmarks, blendshapes), or (None, None) if malformed."""
    if not isinstance(p, (list, tuple)) or len(p) < len(LM_IDS) * 2:
        return None, None
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p):
        return None, None
    lm = {}
    for i, idx in enumerate(LM_IDS):
        lm[idx] = (float(p[2 * i]), float(p[2 * i + 1]))
    blend = None
    if len(p) >= len(LM_IDS) * 2 + len(BLEND_KEYS):
        base = len(LM_IDS) * 2
        blend = {k: float(p[base + i]) for i, k in enumerate(BLEND_KEYS)}
    return lm, blend


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mouth_aperture(lm):
    return dist(lm[IDX["lipTop"]], lm[IDX["lipBottom"]]) / (dist(lm[IDX["forehead"]], lm[IDX["chin"]]) or 1)


def eye_aspect(lm, ids):
    p1, p2, p3, p4, p5, p6 = (lm[i] for i in ids)
    return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4) or 1)


def _trim(seq, now, window_ms):
    while seq and seq[0]["t"] < now - window_ms:
        seq.pop(0)
    return seq


class AudioMetrics:
    """Vocoder signature, prosodic statistics and respiratory coupling."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.first_t = None
        self.last_t = None
        self.f0 = []
        self.cliff = []
        self.speech_ms = []
        self.onsets = []
        self.breaths = []
        self.pause_noise = []
        self._state = "silence"
        self._state_since = 0
        self._pause_noise_ms = 0
        self._last_breath_candidate = False

    def push(self, t, dt, level, flatness, f0, clarity, drop_db):
        if self.first_t is None:
            self.first_t = t
        self.last_t = t
        speaking = level >= SPEECH_RMS

        if speaking:
            self.speech_ms.append({"t": t, "v": dt})
            if f0 > 0 and clarity > 0.6:
                self.f0.append({"t": t, "v": 12 * math.log2(f0 / 100)})
                self.cliff.append({"t": t, "v": 1 if drop_db >= 40 else 0})
        _trim(self.speech_ms, t, 30000)
        _trim(self.f0, t, 12000)
        _trim(self.cliff, t, 10000)

        # Phrase / pause state machine, with breath detection inside pauses.
        if speaking and self._state != "speech":
            if self._state == "pause" and t - self._state_since > 150:
                self.onsets.append(t)
            self._state = "speech"
            self._state_since = t
            self._last_breath_candidate = False
        elif not speaking:
            if self._state != "pause":
                self._state = "pause"
                self._state_since = t
                self._pause_noise_ms = 0
            noisy = 0.0015 <= level < SPEECH_RMS and flatness > 0.25
            self.pause_noise.append({"t": t, "v": 1 if level < DIGITAL_SILENCE else 0})
            if noisy:
                self._pause_noise_ms += dt
                if 120 <= self._pause_noise_ms <= 900 and not self._last_breath_candidate:
                    self._last_breath_candidate = True
                    self.breaths.append(t)
        _trim(self.pause_noise, t, 30000)
        self.onsets = [o for o in self.onsets if o > t - 30000]
        self.breaths = [b for b in self.breaths if b > t - 30000]

    @property
    def speech_seconds(self):
        return sum(s["v"] for s in self.speech_ms) / 1000

    def _coupling(self):
        if not self.onsets:
            return None
        coupled = sum(1 for o in self.onsets if any(0 <= o - b < 900 for b in self.breaths))
        return coupled / len(self.onsets)

    def evaluate(self):
        out = []

        # PROSODY_FLAT: conversational speech usually spans 2 to 4+ semitones of robust spread.
        if len(self.f0) >= 100:
            spread = robust_std([p["v"] for p in self.f0])
            fired = spread < 1.1
            out.append({
                "id": "PROSODY_FLAT",
                "fired": fired,
                "severity": clamp(0.9 - spread * 0.5, 0.35, 0.8) if fired else 0,
                "detail": f"Pitch variation {spread:.1f} semitones"
                          + (": monotone delivery, typical of synthesised speech." if fired else "."),
            })

        # VOCODER_ARTIFACT: spectral cliff above 5 kHz on most voiced frames.
        if len(self.cliff) >= 50:
            frac = sum(c["v"] for c in self.cliff) / len(self.cliff)
            fired = frac > 0.8
            out.append({
                "id": "VOCODER_ARTIFACT",
                "fired": fired,
                "severity": 0.5 if fired else 0,
                "detail": ("Hard spectral cutoff above 5 kHz on most voiced audio, a band-limited vocoder or "
                           "codec signature. Some wideband codecs also do this.")
                          if fired else "No abnormal spectral cutoff.",
            })

        # BREATH_ABSENT: speech with pauses but no breath noise before phrase onsets.
        if self.speech_seconds >= 15 and len(self.onsets) >= 4:
            coupling = self._coupling()
            silent = (sum(p["v"] for p in self.pause_noise) / len(self.pause_noise)) if self.pause_noise else 0
            fired = coupling < 0.1
            out.append({
                "id": "BREATH_ABSENT",
                "fired": fired,
                "severity": (0.7 if silent > 0.8 else 0.45) if fired else 0,
                "detail": (f"No breaths before {len(self.onsets)} phrases"
                           + (" and pauses are digitally silent" if silent > 0.8 else "")
                           + ". Noise suppression can cause this too.")
                          if fired else f"Breaths precede {coupling * 100:.0f}% of phrases.",
            })
        return out

    def probe(self):
        spread = robust_std([p["v"] for p in self.f0]) if len(self.f0) >= 20 else None
        frac = (sum(c["v"] for c in self.cliff) / len(self.cliff)) if len(self.cliff) >= 10 else None
        coupling = self._coupling()
        secs = self.speech_seconds
        return [
            {"key": "speech", "label": "Speech heard", "progress": clamp(secs / 15, 0, 1),
             "text": f"{secs:.1f} s over {len(self.onsets)} phrases", "bar": 0},
            {"key": "pitch", "label": "Pitch variation", "progress": clamp(len(self.f0) / 100, 0, 1),
             "text": "measuring" if spread is None else f"{spread:.1f} semitones",
             "bar": 0 if spread is None else clamp(1 - spread / 2.5, 0, 1)},
            {"key": "cliff", "label": "Spectral cutoff", "progress": clamp(len(self.cliff) / 50, 0, 1),
             "text": "measuring" if frac is None else f"{frac * 100:.0f}% of voiced frames cut off above 5 kHz",
             "bar": 0 if frac is None else clamp(frac, 0, 1)},
            {"key": "breath", "label": "Breaths before phrases",
             "progress": min(clamp(secs / 15, 0, 1), clamp(len(self.onsets) / 4, 0, 1)),
             "text": "measuring" if coupling is None else f"{coupling * 100:.0f}% of phrases preceded by a breath",
             "bar": 0 if coupling is None else clamp(1 - coupling / 0.4, 0, 1)},
        ]


class VideoMetrics:
    """Blink cadence, motion smoothness and expression variance from face landmarks."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.first_t = None
        self.last_t = None
        self.ear_hist = []
        self.closed_at = None
        self.blinks = []
        self.pose = []
        self.blend = []
        self.mouth = []
        self.frames = 0

    def push(self, t, lm, blend):
        if not lm:
            return
        self.frames += 1
        if self.first_t is None:
            self.first_t = t
        self.last_t = t

        self.mouth.append({"t": t, "v": mouth_aperture(lm)})
        _trim(self.mouth, t, 20000)

        ear = (eye_aspect(lm, IDX["rightEye"]) + eye_aspect(lm, IDX["leftEye"])) / 2
        self.ear_hist.append({"t": t, "v": ear})
        _trim(self.ear_hist, t, 15000)
        self._blink_step(t, ear)

        iod = dist(lm[33], lm[263]) or 1
        stable = IDX["stable"]
        cx = sum(lm[i][0] for i in stable) / len(stable)
        cy = sum(lm[i][1] for i in stable) / len(stable)
        v = []
        for i in stable:
            v.append((lm[i][0] - cx) / iod)
            v.append((lm[i][1] - cy) / iod)
        self.pose.append({"t": t, "v": v})
        _trim(self.pose, t, 6000)

        if blend:
            self.blend.append({"t": t, "v": [blend.get(k, 0) for k in BLEND_KEYS]})
            _trim(self.blend, t, 12000)
        self.blinks = [b for b in self.blinks if b > t - 60000]

    def _blink_step(self, t, ear):
        if len(self.ear_hist) < 20:
            return
        open_level = median([e["v"] for e in self.ear_hist if e["v"] > 0])
        thr = open_level * 0.72
        if ear < thr:
            if self.closed_at is None:
                self.closed_at = t
        elif self.closed_at is not None:
            d = t - self.closed_at
            if 40 <= d <= 600:
                self.blinks.append(self.closed_at)
            self.closed_at = None

    @property
    def observed_ms(self):
        return 0 if self.first_t is None else self.last_t - self.first_t

    def blink_rate(self, now=None):
        now = self.last_t if now is None else now
        win = min(self.observed_ms, 45000)
        if win < 1000:
            return 0.0
        n = sum(1 for b in self.blinks if b >= now - win)
        return (n / win) * 60000

    def blink_interval_cv(self):
        if len(self.blinks) < 7:
            return None
        iv = [self.blinks[i] - self.blinks[i - 1] for i in range(1, len(self.blinks))]
        m = sum(iv) / len(iv)
        return std(iv) / (m or 1)

    def jitter_stats(self):
        """Flicker of facial geometry relative to head pose.

        Only consecutive, regularly spaced frames are used: the tracker returns nothing when the video frame
        has not advanced, and a second difference taken across a dropped frame reads as a large spike, which
        on its own fired MOTION_SMOOTHNESS constantly.
        """
        p = self.pose
        if len(p) < 45:
            return None
        dts = [p[i]["t"] - p[i - 1]["t"] for i in range(1, len(p))]
        md = median(dts)
        if md <= 0:
            return None
        k = len(p[0]["v"])
        jitter = vel = 0.0
        n = 0
        for i in range(2, len(p)):
            gap_a = p[i - 1]["t"] - p[i - 2]["t"]
            gap_b = p[i]["t"] - p[i - 1]["t"]
            if not (md * 0.6 <= gap_a <= md * 1.6 and md * 0.6 <= gap_b <= md * 1.6):
                continue
            a, b, c = p[i]["v"], p[i - 1]["v"], p[i - 2]["v"]
            for j in range(k):
                d2 = a[j] - 2 * b[j] + c[j]
                jitter += d2 * d2
                d1 = a[j] - b[j]
                vel += d1 * d1
            n += 1
        if n < 30:
            return None
        jr = math.sqrt(jitter / n / k)
        vr = math.sqrt(vel / n / k)
        return {"jr": jr, "vr": vr, "ratio": jr / vr if vr > 0 else 0,
                "n": n, "dropped": sum(1 for d in dts if d > md * 1.6), "fps": 1000 / md}

    def _expression_variance(self):
        if len(self.blend) < 20:
            return None
        cols = list(zip(*[b["v"] for b in self.blend]))
        sds = [std(c) for c in cols]
        return sum(sds) / len(sds)

    def evaluate(self):
        out = []
        obs = self.observed_ms

        if obs >= 25000:
            rate = self.blink_rate()
            cv = self.blink_interval_cv()
            fired, severity = False, 0
            detail = f"{rate:.0f} blinks/min" + (f", interval variation {cv:.2f}" if cv is not None else "") + "."
            if rate < 5:
                fired, severity = True, clamp(0.5 + (5 - rate) / 10, 0.5, 0.9)
                detail += " Far fewer blinks than a person normally makes."
            elif rate > 40:
                fired, severity = True, 0.5
                detail += " Unusually frequent blinking."
            elif cv is not None and cv < 0.18:
                fired, severity = True, 0.6
                detail += " Blinks are metronomic; real blinking is irregular."
            out.append({"id": "BLINK_CADENCE", "fired": fired, "severity": severity, "detail": detail})

        js = self.jitter_stats()
        if js and obs >= 5000:
            jr, vr, ratio = js["jr"], js["vr"], js["ratio"]
            fired, severity = False, 0
            detail = (f"Landmark jitter {jr * 1000:.1f}, motion {vr * 1000:.1f}, "
                      f"noise ratio {ratio:.2f} (per-mille of eye distance).")
            if jr > JITTER_FLOOR and ratio > NOISE_RATIO and vr > MIN_MOTION:
                fired, severity = True, clamp(0.4 + (jr - JITTER_FLOOR) * 20, 0.4, 0.9)
                detail += " Facial geometry flickers between frames."
            elif vr < 0.0003 and obs > 8000:
                fired, severity = True, 0.35
                detail += " Face is unnaturally still."
            out.append({"id": "MOTION_SMOOTHNESS", "fired": fired, "severity": severity, "detail": detail})

        if len(self.blend) >= 150 and self.blend[-1]["t"] - self.blend[0]["t"] >= 9000:
            avg = self._expression_variance()
            fired = avg < 0.008
            out.append({
                "id": "EXPRESSION_FLAT",
                "fired": fired,
                "severity": clamp(0.8 - avg * 50, 0.3, 0.7) if fired else 0,
                "detail": f"Expression variation {avg * 100:.2f}%"
                          + (": the face barely changes expression." if fired else "."),
            })
        return out

    def probe(self):
        obs = self.observed_ms
        js = self.jitter_stats()
        rate = self.blink_rate()
        cv = self.blink_interval_cv()
        blend_ms = (self.blend[-1]["t"] - self.blend[0]["t"]) if self.blend else 0
        expr = self._expression_variance()
        return [
            {"key": "face", "label": "Face tracking", "progress": 1 if self.frames else 0,
             "text": (f"{self.frames} frames" + (f" at {js['fps']:.0f} fps, {js['dropped']} gaps" if js else ""))
                     if self.frames else "waiting for a face",
             "bar": 0},
            {"key": "blink", "label": "Blink cadence", "progress": clamp(obs / 25000, 0, 1),
             "text": (f"{rate:.0f} blinks/min" + (f", variation {cv:.2f}" if cv is not None else ""))
                     if obs >= 25000 else f"{len(self.blinks)} blinks so far",
             "bar": clamp(1 - rate / 18, 0, 1)},
            {"key": "jitter", "label": "Landmark jitter",
             "progress": 1 if js else clamp(len(self.pose) / 45, 0, 1),
             "text": (f"{js['jr'] * 1000:.1f} jitter vs {js['vr'] * 1000:.1f} motion, ratio {js['ratio']:.2f}")
                     if js else "measuring",
             "bar": clamp(js["jr"] / (JITTER_FLOOR * 2), 0, 1) if js else 0},
            {"key": "expr", "label": "Expression variance",
             "progress": min(clamp(len(self.blend) / 150, 0, 1), clamp(blend_ms / 9000, 0, 1)),
             "text": "measuring" if expr is None else f"{expr * 100:.2f}% average movement",
             "bar": 0 if expr is None else clamp(1 - expr / 0.02, 0, 1)},
        ]


def resample(track, hz, t0, t1):
    """Samples an irregular [{t, v}] track onto a uniform grid; None where the track does not cover it."""
    out = []
    if len(track) < 2:
        return out
    step = 1000 / hz
    j = 0
    t = t0
    while t <= t1:
        while j < len(track) - 2 and track[j + 1]["t"] < t:
            j += 1
        a, b = track[j], track[j + 1]
        if t < a["t"] or t > b["t"]:
            out.append(None)
        else:
            f = 0 if b["t"] == a["t"] else (t - a["t"]) / (b["t"] - a["t"])
            out.append(a["v"] + (b["v"] - a["v"]) * f)
        t += step
    return out


def moving_average(arr, n):
    if n <= 1:
        return list(arr)
    out = []
    total = 0.0
    for i, v in enumerate(arr):
        total += v
        if i >= n:
            total -= arr[i - n]
        out.append(total / min(n, i + 1))
    return out


def zscore(arr):
    a = np.asarray(arr, dtype=float)
    s = std(a) or 1
    return (a - a.mean()) / s


def cross_correlate(x, y, max_lag):
    """Positive lag means y trails x: with x = mouth and y = audio, audio arrives late."""
    n = min(len(x), len(y))
    zx, zy = zscore(x[:n]), zscore(y[:n])
    best_lag, best_corr = 0, -math.inf
    for k in range(-max_lag, max_lag + 1):
        lo, hi = max(0, -k), min(n, n - k)
        c = hi - lo
        r = float(np.dot(zx[lo:hi], zy[lo + k : hi + k]) / c) if c > n / 2 else 0.0
        if r > best_corr:
            best_lag, best_corr = k, r
    return best_lag, best_corr


class CrossModal:
    """Mouth aperture versus audio energy envelope on a shared clock: the lag is the sync offset."""

    def __init__(self, window_ms=8000, hz=25, max_lag_ms=500, drift_ms=150):
        self.window_ms = window_ms
        self.hz = hz
        self.max_lag_ms = max_lag_ms
        self.drift_ms = drift_ms
        self.reset()

    def reset(self):
        self.mouth = []
        self.audio = []
        self.lag_history = []

    def push_mouth(self, t, v):
        self.mouth.append({"t": t, "v": v})
        _trim(self.mouth, t, self.window_ms + 2000)

    def push_audio(self, t, v):
        self.audio.append({"t": t, "v": v})
        _trim(self.audio, t, self.window_ms + 2000)

    def measure(self, now):
        if not self.mouth or not self.audio:
            return None
        # End the window at the newest sample both tracks have; sampling to `now` would run past the data.
        t1 = min(self.mouth[-1]["t"], self.audio[-1]["t"])
        if now - t1 > 1500:
            return None
        t0 = t1 - self.window_ms
        m = resample(self.mouth, self.hz, t0, t1)
        a = resample(self.audio, self.hz, t0, t1)
        if len(m) < self.hz * 5 or any(v is None for v in m) or any(v is None for v in a):
            return None
        ms = moving_average(m, 2)
        a_s = moving_average(a, 3)
        if std(ms) < 0.004 or std(a_s) < 0.004:
            return None
        max_lag = round((self.max_lag_ms / 1000) * self.hz)
        lag, corr = cross_correlate(ms, a_s, max_lag)
        return {"lag_ms": (lag / self.hz) * 1000, "peak": corr}

    def evaluate(self, now):
        r = self.measure(now)
        if not r:
            return []
        self.lag_history.append(r["lag_ms"])
        if len(self.lag_history) > 8:
            self.lag_history.pop(0)

        rng = (max(self.lag_history) - min(self.lag_history)) if len(self.lag_history) >= 5 else 0
        offset = abs(r["lag_ms"]) >= self.drift_ms and r["peak"] >= 0.25
        drifting = rng >= self.drift_ms and r["peak"] >= 0.2
        fired = offset or drifting
        return [
            {"id": "LIP_DRIFT", "fired": fired,
             "severity": clamp(max(abs(r["lag_ms"]), rng) / 400, 0.4, 1) if fired else 0,
             "detail": f"Audio {'lags' if r['lag_ms'] >= 0 else 'leads'} the mouth by "
                       f"{abs(r['lag_ms']):.0f} ms (peak correlation {r['peak']:.2f}"
                       + (f", lag wandered {rng:.0f} ms" if drifting else "") + ")."},
            {"id": "LIP_DECOUPLED", "fired": r["peak"] < 0.12,
             "severity": 0.5 if r["peak"] < 0.12 else 0,
             "detail": (f"Lip movement and voice barely correlate ({r['peak']:.2f})." if r["peak"] < 0.12
                        else f"Lips and voice correlate ({r['peak']:.2f}).")},
        ]

    def probe(self, now):
        r = self.measure(now)
        has_both = bool(self.mouth and self.audio)
        return [
            {"key": "sync", "label": "Lip-sync offset", "progress": 1 if r else (0.6 if has_both else 0),
             "text": (f"{'audio behind' if r['lag_ms'] >= 0 else 'audio ahead'} {abs(r['lag_ms']):.0f} ms")
                     if r else "needs a speaking face and audible voice",
             "bar": clamp(abs(r["lag_ms"]) / 400, 0, 1) if r else 0},
            {"key": "corr", "label": "Lip/voice correlation", "progress": 1 if r else 0,
             "text": f"{r['peak']:.2f}" if r else "measuring",
             "bar": clamp(1 - r["peak"] / 0.5, 0, 1) if r else 0},
        ]
