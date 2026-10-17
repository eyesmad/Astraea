import math
import numpy as np
from scipy import signal

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def trim_window(arr, now, window_ms):
    limit = now - window_ms
    idx = 0
    while idx < len(arr) and arr[idx]['t'] < limit:
        idx += 1
    if idx > 0:
        del arr[:idx]

# --- DSP ---
def spectrum_db(frame):
    n = len(frame)
    w = np.blackman(n)
    sp = np.fft.rfft(frame * w)
    mag = np.abs(sp[:-1]) / n
    mag = np.maximum(mag, 1e-12)
    return np.maximum(-120.0, 20 * np.log10(mag))

# --- Audio Features ---
def rms(buf):
    if len(buf) == 0: return 0.0
    return float(np.sqrt(np.mean(buf**2)))

def spectral_flatness(freq_db, lo_bin, hi_bin):
    n = hi_bin - lo_bin
    if n <= 0: return 0.0
    p = 10 ** (freq_db[lo_bin:hi_bin] / 10.0) + 1e-12
    log_mean = np.mean(np.log(p))
    arith_mean = np.mean(p)
    return float(np.exp(log_mean) / arith_mean)

def estimate_f0(buf, sample_rate, fmin=70, fmax=400):
    min_lag = int(sample_rate / fmax)
    max_lag = min(int(sample_rate / fmin), len(buf) - 1)
    n = len(buf) - max_lag
    if n <= 0: return {"f0": 0.0, "clarity": 0.0}
    
    e0 = np.sum(buf[:n]**2)
    if e0 < 1e-9: return {"f0": 0.0, "clarity": 0.0}
    
    r = np.zeros(max_lag + 2, dtype=np.float32)
    for lag in range(min_lag, max_lag + 1):
        s = np.sum(buf[:n] * buf[lag:n+lag])
        e1 = np.sum(buf[lag:n+lag]**2)
        r[lag] = s / np.sqrt(e0 * e1 + 1e-12)
        
    thresh = 0.85 * np.max(r[min_lag:max_lag+1])
    best = 0.0
    best_lag = 0
    for lag in range(min_lag + 1, max_lag):
        if r[lag] >= thresh and r[lag] >= r[lag-1] and r[lag] >= r[lag+1]:
            best = r[lag]
            best_lag = lag
            break
            
    if not best_lag: return {"f0": 0.0, "clarity": 0.0}
    return {"f0": float(sample_rate / best_lag), "clarity": float(best)}

def brickwall_drop(freq_db, sample_rate, fft_size):
    bin_hz = sample_rate / fft_size
    band_bins = round(500 / bin_hz)
    bands = []
    start = 0
    while start + band_bins <= len(freq_db):
        s = np.mean(freq_db[start:start+band_bins])
        bands.append({"hz": start * bin_hz, "db": s})
        start += band_bins
        
    best = {"dropDb": 0.0, "atHz": 0.0}
    for i in range(1, len(bands) - 2):
        if bands[i]["hz"] < 5000: continue
        drop = bands[i-1]["db"] - bands[i]["db"]
        tail = max(bands[i+1]["db"], bands[i+2]["db"])
        if drop > best["dropDb"] and bands[i-1]["db"] - tail > drop * 0.9:
            best = {"dropDb": float(drop), "atHz": float(bands[i]["hz"])}
    return best

# --- Signals ---
SIGNALS = {
    'CALLER_AUTH': {'layer': 'metadata', 'label': 'STIR/SHAKEN attestation', 'weight': 0.7, 'standing': True},
    'NUMBER_REPUTATION': {'layer': 'metadata', 'label': 'Number reputation', 'weight': 0.6, 'standing': True},
    'GEO_MISMATCH': {'layer': 'metadata', 'label': 'Geographic mismatch', 'weight': 0.6, 'standing': True},
    'LINE_TYPE': {'layer': 'metadata', 'label': 'Line type', 'weight': 0.4, 'standing': True},
    'AREA_CODE': {'layer': 'metadata', 'label': 'Area code', 'weight': 0.4, 'standing': True},
    'BLINK_CADENCE': {'layer': 'video', 'label': 'Blink cadence', 'weight': 0.5},
    'MOTION_SMOOTHNESS': {'layer': 'video', 'label': 'Motion smoothness', 'weight': 0.4},
    'EXPRESSION_FLAT': {'layer': 'video', 'label': 'Expression variance', 'weight': 0.3},
    'VOCODER_ARTIFACT': {'layer': 'voice', 'label': 'Vocoder signature', 'weight': 0.5},
    'PROSODY_FLAT': {'layer': 'voice', 'label': 'Prosody', 'weight': 0.4},
    'BREATH_ABSENT': {'layer': 'voice', 'label': 'Respiratory coupling', 'weight': 0.4},
    'VOICE_MODEL': {'layer': 'voice', 'label': 'Voice model', 'weight': 0.6},
    'LIP_DRIFT': {'layer': 'crossmodal', 'label': 'Lip-sync drift', 'weight': 0.7},
    'LIP_DECOUPLED': {'layer': 'crossmodal', 'label': 'Lips decoupled from voice', 'weight': 0.5},
    'SCAM_CONTENT': {'layer': 'content', 'label': 'Scam script match', 'weight': 0.7, 'standing': True},
    'CHALLENGE_FAILED': {'layer': 'challenge', 'label': 'Challenge failed', 'weight': 0.9, 'standing': True},
}
SYNERGY = [
    ['LIP_DRIFT', 'BLINK_CADENCE'],
    ['LIP_DRIFT', 'MOTION_SMOOTHNESS'],
    ['VOCODER_ARTIFACT', 'LIP_DRIFT'],
    ['PROSODY_FLAT', 'BREATH_ABSENT'],
    ['VOICE_MODEL', 'LIP_DRIFT'],
]

# --- AudioMetrics ---
SPEECH_RMS = 0.02
DIGITAL_SILENCE = 2e-4

class AudioMetrics:
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
        self._state = 'silence'
        self._state_since = 0
        self._pause_noise_ms = 0
        self._pause_frames = 0
        self._pause_silent_frames = 0
        self._last_breath_candidate = False

    def push(self, f):
        t = f['t']
        if self.first_t is None: self.first_t = t
        self.last_t = t
        speaking = f['rms'] >= SPEECH_RMS

        if speaking:
            self.speech_ms.append({'t': t, 'v': f['dt']})
            if f['f0'] > 0 and f['clarity'] > 0.6:
                self.f0.append({'t': t, 'v': 12 * math.log2(f['f0'] / 100)})
                self.cliff.append({'t': t, 'v': 1 if f['dropDb'] >= 40 else 0})
        
        trim_window(self.speech_ms, t, 30000)
        trim_window(self.f0, t, 12000)
        trim_window(self.cliff, t, 10000)

        if speaking and self._state != 'speech':
            if self._state == 'pause' and t - self._state_since > 150:
                self.onsets.append(t)
            self._state = 'speech'
            self._state_since = t
            self._last_breath_candidate = False
        elif not speaking:
            if self._state != 'pause':
                self._state = 'pause'
                self._state_since = t
                self._pause_noise_ms = 0
            noisy = f['rms'] >= 0.0015 and f['rms'] < SPEECH_RMS and f['flatness'] > 0.25
            self._pause_frames += 1
            if f['rms'] < DIGITAL_SILENCE:
                self._pause_silent_frames += 1
            self.pause_noise.append({'t': t, 'v': 1 if f['rms'] < DIGITAL_SILENCE else 0})
            if noisy:
                self._pause_noise_ms += f['dt']
                if 120 <= self._pause_noise_ms <= 900 and not self._last_breath_candidate:
                    self._last_breath_candidate = True
                    self.breaths.append(t)
        
        trim_window(self.pause_noise, t, 30000)
        self.onsets = [o for o in self.onsets if o > t - 30000]
        self.breaths = [b for b in self.breaths if b > t - 30000]

    @property
    def speech_seconds(self):
        return sum(s['v'] for s in self.speech_ms) / 1000.0

    def evaluate(self):
        out = []
        if len(self.f0) >= 100:
            vals = [p['v'] for p in self.f0]
            spread = (np.percentile(vals, 75) - np.percentile(vals, 25)) / 1.349 if len(vals) > 0 else 0
            fired = spread < 1.1
            out.append({
                'id': 'PROSODY_FLAT',
                'fired': fired,
                'severity': clamp(0.9 - spread * 0.5, 0.35, 0.8) if fired else 0,
                'detail': f"Pitch variation {spread:.1f} semitones{': monotone delivery, typical of synthesised speech.' if fired else '.'}"
            })

        if len(self.cliff) >= 50:
            frac = sum(c['v'] for c in self.cliff) / len(self.cliff)
            fired = frac > 0.8
            out.append({
                'id': 'VOCODER_ARTIFACT',
                'fired': fired,
                'severity': 0.5 if fired else 0,
                'detail': 'Hard spectral cutoff above 5 kHz on most voiced audio, a band-limited vocoder or codec signature. Some wideband codecs also do this.' if fired else 'No abnormal spectral cutoff.'
            })

        if self.speech_seconds >= 15 and len(self.onsets) >= 4:
            coupled = sum(1 for o in self.onsets if any(0 <= (o - b) < 900 for b in self.breaths))
            coupling = coupled / len(self.onsets)
            silent = sum(p['v'] for p in self.pause_noise) / len(self.pause_noise) if self.pause_noise else 0
            fired = coupling < 0.1
            out.append({
                'id': 'BREATH_ABSENT',
                'fired': fired,
                'severity': (0.7 if silent > 0.8 else 0.45) if fired else 0,
                'detail': f"No breaths before {len(self.onsets)} phrases{' and pauses are digitally silent' if silent > 0.8 else ''}. Noise suppression can cause this too." if fired else f"Breaths precede {coupling * 100:.0f}% of phrases."
            })
        return out

    def probe(self):
        vals = [p['v'] for p in self.f0]
        spread = (np.percentile(vals, 75) - np.percentile(vals, 25)) / 1.349 if len(self.f0) >= 20 else None
        frac = sum(c['v'] for c in self.cliff) / len(self.cliff) if len(self.cliff) >= 10 else None
        coupled = sum(1 for o in self.onsets if any(0 <= (o - b) < 900 for b in self.breaths))
        coupling = coupled / len(self.onsets) if len(self.onsets) > 0 else None
        secs = self.speech_seconds
        
        return [
            {
                'key': 'speech', 'label': 'Speech heard', 'progress': clamp(secs / 15.0, 0, 1),
                'text': f"{secs:.1f} s over {len(self.onsets)} phrases",
                'bar': 0
            },
            {
                'key': 'pitch', 'label': 'Pitch variation', 'progress': clamp(len(self.f0) / 100.0, 0, 1),
                'text': f"{spread:.1f} semitones" if spread is not None else "measuring",
                'bar': clamp(1 - spread / 2.5, 0, 1) if spread is not None else 0
            },
            {
                'key': 'cliff', 'label': 'Spectral cutoff', 'progress': clamp(len(self.cliff) / 50.0, 0, 1),
                'text': f"{frac * 100:.0f}% of voiced frames cut off above 5 kHz" if frac is not None else "measuring",
                'bar': clamp(frac, 0, 1) if frac is not None else 0
            },
            {
                'key': 'breath', 'label': 'Breaths before phrases', 'progress': min(clamp(secs / 15.0, 0, 1), clamp(len(self.onsets) / 4.0, 0, 1)),
                'text': f"{coupling * 100:.0f}% of phrases preceded by a breath" if coupling is not None else "measuring",
                'bar': clamp(1 - coupling / 0.4, 0, 1) if coupling is not None else 0
            }
        ]

# --- VideoMetrics ---
IDX_STABLE = [1, 152, 33, 263, 61, 291, 10]
IDX_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
IDX_LEFT_EYE = [362, 385, 387, 263, 373, 380]
LM_IDS = []
for idx in IDX_RIGHT_EYE + IDX_LEFT_EYE + [1, 10, 152, 13, 14] + IDX_STABLE:
    if idx not in LM_IDS:
        LM_IDS.append(idx)
BLEND_KEYS = [
    'jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'browDownLeft', 'browDownRight',
    'browInnerUp', 'eyeSquintLeft', 'eyeSquintRight', 'mouthFrownLeft', 'mouthFrownRight', 'cheekPuff', 'mouthPucker'
]

def unpack_face(p):
    if not isinstance(p, list) or len(p) < len(LM_IDS) * 2:
        return None
    lm = {}
    for i, idx in enumerate(LM_IDS):
        lm[idx] = {'x': p[2 * i], 'y': p[2 * i + 1]}
    blend = None
    if len(p) >= len(LM_IDS) * 2 + len(BLEND_KEYS):
        blend = p[len(LM_IDS) * 2:]
    return {'lm': lm, 'blend': blend}

JITTER_FLOOR = 0.025
NOISE_RATIO = 1.55
MIN_MOTION = 0.004

def dist(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])

def eye_aspect(lm, ids):
    p1, p2, p3, p4, p5, p6 = (lm[i] for i in ids)
    return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4) or 1)

def mouth_aperture(lm):
    return dist(lm[13], lm[14]) / (dist(lm[10], lm[152]) or 1)

class VideoMetrics:
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
        if not lm: return
        self.frames += 1
        if self.first_t is None: self.first_t = t
        self.last_t = t

        self.mouth.append({'t': t, 'v': mouth_aperture(lm)})
        trim_window(self.mouth, t, 20000)

        ear = (eye_aspect(lm, IDX_RIGHT_EYE) + eye_aspect(lm, IDX_LEFT_EYE)) / 2
        self.ear_hist.append({'t': t, 'v': ear})
        trim_window(self.ear_hist, t, 15000)
        self._blink_step(t, ear)

        iod = dist(lm[33], lm[263]) or 1
        cx = sum(lm[i]['x'] for i in IDX_STABLE) / len(IDX_STABLE)
        cy = sum(lm[i]['y'] for i in IDX_STABLE) / len(IDX_STABLE)
        v = []
        for i in IDX_STABLE:
            v.extend([(lm[i]['x'] - cx) / iod, (lm[i]['y'] - cy) / iod])
        self.pose.append({'t': t, 'v': v})
        trim_window(self.pose, t, 6000)

        if blend:
            # We assume blend is a list of numbers matching BLEND_KEYS length
            # The JS code mapped object keys. Python side we should assume a list.
            self.blend.append({'t': t, 'v': blend})
            trim_window(self.blend, t, 12000)
            
        self.blinks = [b for b in self.blinks if b > t - 60000]

    def _blink_step(self, t, ear):
        if len(self.ear_hist) < 20: return
        open_vals = [e['v'] for e in self.ear_hist if e['v'] > 0]
        if not open_vals: return
        open_val = np.median(open_vals)
        # Increased from 0.72 to 0.82 to be more sensitive to shallow blinks
        thr = open_val * 0.82
        if ear < thr:
            if self.closed_at is None: self.closed_at = t
        elif self.closed_at is not None:
            d = t - self.closed_at
            # Lowered min duration from 40 to 30 to catch fast 1-frame blinks
            if 30 <= d <= 600:
                self.blinks.append(self.closed_at)
            self.closed_at = None

    @property
    def observed_ms(self):
        return 0 if self.first_t is None else self.last_t - self.first_t

    def blink_rate(self, now=None):
        if now is None: now = self.last_t
        win = min(self.observed_ms, 45000)
        if win < 1000: return 0.0
        n = sum(1 for b in self.blinks if b >= now - win)
        return (n / win) * 60000.0

    def blink_interval_cv(self):
        if len(self.blinks) < 7: return None
        iv = [self.blinks[i] - self.blinks[i-1] for i in range(1, len(self.blinks))]
        m = np.mean(iv)
        return np.std(iv, ddof=1) / m if m else 0.0

    def jitter_stats(self):
        if len(self.pose) < 45: return None
        dts = [self.pose[i]['t'] - self.pose[i-1]['t'] for i in range(1, len(self.pose))]
        md = np.median(dts)
        if not (md > 0): return None
        k = len(self.pose[0]['v'])
        jitter = 0.0
        vel = 0.0
        n = 0
        for i in range(2, len(self.pose)):
            gap_a = self.pose[i-1]['t'] - self.pose[i-2]['t']
            gap_b = self.pose[i]['t'] - self.pose[i-1]['t']
            if not (md * 0.6 <= gap_a <= md * 1.6) or not (md * 0.6 <= gap_b <= md * 1.6):
                continue
            for c in range(k):
                d2 = self.pose[i]['v'][c] - 2 * self.pose[i-1]['v'][c] + self.pose[i-2]['v'][c]
                jitter += d2 * d2
                d1 = self.pose[i]['v'][c] - self.pose[i-1]['v'][c]
                vel += d1 * d1
            n += 1
        if n < 30: return None
        jr = math.sqrt(jitter / n / k)
        vr = math.sqrt(vel / n / k)
        ratio = jr / vr if vr > 0 else 0.0
        return {'jr': jr, 'vr': vr, 'ratio': ratio, 'n': n, 'dropped': sum(1 for d in dts if d > md * 1.6), 'fps': 1000.0 / md}

    def evaluate(self):
        out = []
        obs = self.observed_ms

        if obs >= 25000:
            rate = self.blink_rate()
            cv = self.blink_interval_cv()
            fired = False
            severity = 0.0
            detail = f"{rate:.0f} blinks/min{f', interval variation {cv:.2f}' if cv is not None else ''}."
            if rate < 5:
                fired = True
                severity = clamp(0.5 + (5 - rate) / 10.0, 0.5, 0.9)
                detail += " Far fewer blinks than a person normally makes."
            elif rate > 40:
                fired = True
                severity = 0.5
                detail += " Unusually frequent blinking."
            elif cv is not None and cv < 0.18:
                fired = True
                severity = 0.6
                detail += " Blinks are metronomic; real blinking is irregular."
            out.append({'id': 'BLINK_CADENCE', 'fired': fired, 'severity': severity, 'detail': detail})

        js = self.jitter_stats()
        if js and obs >= 5000:
            jr, vr, ratio = js['jr'], js['vr'], js['ratio']
            fired = False
            severity = 0.0
            detail = f"Landmark jitter {jr*1000:.1f}, motion {vr*1000:.1f}, noise ratio {ratio:.2f} (per-mille of eye distance)."
            if jr > JITTER_FLOOR and ratio > NOISE_RATIO and vr > MIN_MOTION:
                fired = True
                severity = clamp(0.4 + (jr - JITTER_FLOOR) * 20.0, 0.4, 0.9)
                detail += " Facial geometry flickers between frames."
            elif vr < 0.0003 and obs > 8000:
                fired = True
                severity = 0.35
                detail += " Face is unnaturally still."
            out.append({'id': 'MOTION_SMOOTHNESS', 'fired': fired, 'severity': severity, 'detail': detail})

        if len(self.blend) >= 150 and self.blend[-1]['t'] - self.blend[0]['t'] >= 9000:
            k = len(self.blend[0]['v'])
            sds = [np.std([b['v'][c] for b in self.blend], ddof=1) for c in range(k)]
            avg = np.mean(sds)
            fired = avg < 0.008
            out.append({
                'id': 'EXPRESSION_FLAT',
                'fired': fired,
                'severity': clamp(0.8 - avg * 50.0, 0.3, 0.7) if fired else 0.0,
                'detail': f"Expression variation {avg*100:.2f}%{': the face barely changes expression.' if fired else '.'}"
            })
            
        return out

    def probe(self):
        obs = self.observed_ms
        js = self.jitter_stats()
        rate = self.blink_rate()
        cv = self.blink_interval_cv()
        blend_ms = self.blend[-1]['t'] - self.blend[0]['t'] if self.blend else 0
        expr = None
        if len(self.blend) >= 20:
            k = len(self.blend[0]['v'])
            sds = [np.std([b['v'][c] for b in self.blend], ddof=1) for c in range(k)]
            expr = np.mean(sds)

        return [
            {
                'key': 'face', 'label': 'Face tracking', 'progress': 1 if self.frames else 0,
                'text': f"{self.frames} frames" + (f" at {js['fps']:.0f} fps, {js['dropped']} gaps" if js else "") if self.frames else "waiting for a face",
                'bar': 0
            },
            {
                'key': 'blink', 'label': 'Blink cadence', 'progress': clamp(obs / 25000.0, 0, 1),
                'text': f"{rate:.0f} blinks/min" + (f", variation {cv:.2f}" if cv is not None else "") if obs >= 25000 else f"{len(self.blinks)} blinks so far",
                'bar': clamp(1 - rate / 18.0, 0, 1)
            },
            {
                'key': 'jitter', 'label': 'Landmark jitter', 'progress': 1 if js else clamp(len(self.pose) / 45.0, 0, 1),
                'text': f"{js['jr']*1000:.1f} jitter vs {js['vr']*1000:.1f} motion, ratio {js['ratio']:.2f}" if js else "measuring",
                'bar': clamp(js['jr'] / (JITTER_FLOOR * 2), 0, 1) if js else 0
            },
            {
                'key': 'expr', 'label': 'Expression variance', 'progress': min(clamp(len(self.blend) / 150.0, 0, 1), clamp(blend_ms / 9000.0, 0, 1)),
                'text': f"{expr*100:.2f}% average movement" if expr is not None else "measuring",
                'bar': clamp(1 - expr / 0.02, 0, 1) if expr is not None else 0
            }
        ]

# --- CrossModal ---
def moving_average(arr, n):
    if n <= 1: return np.copy(arr)
    out = np.zeros(len(arr), dtype=float)
    s = 0.0
    for i in range(len(arr)):
        s += arr[i]
        if i >= n:
            s -= arr[i - n]
        out[i] = s / min(n, i + 1)
    return out

def zscore(a):
    if len(a) == 0: return a
    m = np.mean(a)
    s = np.std(a, ddof=1)
    if s == 0: s = 1.0
    return (a - m) / s

def resample(arr, hz, t0, t1):
    out = []
    if len(arr) < 2: return out
    step = 1000.0 / hz
    j = 0
    t = t0
    # Use 1e-9 tolerance for float matching JS '<='
    while t <= t1 + 1e-9:
        while j < len(arr) - 2 and arr[j + 1]['t'] < t:
            j += 1
        a = arr[j]
        b = arr[j + 1]
        if t < a['t'] or t > b['t']:
            out.append(None)
        else:
            f = 0.0 if b['t'] == a['t'] else (t - a['t']) / (b['t'] - a['t'])
            out.append(a['v'] + (b['v'] - a['v']) * f)
        t += step
    return out

def cross_correlate(x, y, max_lag):
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    
    best_lag = 0
    best_corr = -float('inf')
    
    for k in range(-max_lag, max_lag + 1):
        if k < 0:
            x_slice = x[-k:]
            y_slice = y[:n+k]
        else:
            x_slice = x[:n-k]
            y_slice = y[k:]
            
        c = len(x_slice)
        if c > n / 2:
            r = np.dot(x_slice, y_slice) / c
        else:
            r = 0.0
            
        if r > best_corr:
            best_corr = r
            best_lag = k
            
    return {'lag': best_lag, 'corr': float(best_corr) if best_corr != -float('inf') else 0.0}

class CrossModal:
    def __init__(self, window_ms=8000, hz=25, max_lag_ms=500, drift_ms=100):
        self.window_ms = window_ms
        self.hz = hz
        self.max_lag_ms = max_lag_ms
        self.drift_ms = drift_ms
        self.reset()

    def reset(self):
        self.mouth = []
        self.audio = []
        self.lag_history = []
        self.last = None

    def push_mouth(self, t, v):
        self.mouth.append({'t': t, 'v': v})
        trim_window(self.mouth, t, self.window_ms + 2000)

    def push_audio(self, t, v):
        self.audio.append({'t': t, 'v': v})
        trim_window(self.audio, t, self.window_ms + 2000)

    def measure(self, now):
        if not self.mouth or not self.audio: return None
        t1 = min(self.mouth[-1]['t'], self.audio[-1]['t'])
        if now - t1 > 1500: return None
        t0 = t1 - self.window_ms
        m = resample(self.mouth, self.hz, t0, t1)
        a = resample(self.audio, self.hz, t0, t1)
        if len(m) < self.hz * 5 or len(a) < self.hz * 5: return None
        if any(v is None for v in m) or any(v is None for v in a): return None
        
        m_arr = np.array(m, dtype=float)
        a_arr = np.array(a, dtype=float)
        ms = moving_average(m_arr, 2)
        as_arr = moving_average(a_arr, 3)
        if np.std(ms, ddof=1) < 0.004 or np.std(as_arr, ddof=1) < 0.004: return None
        
        max_lag = round((self.max_lag_ms / 1000.0) * self.hz)
        res = cross_correlate(zscore(ms), zscore(as_arr), max_lag)
        return {'lagMs': (res['lag'] / self.hz) * 1000.0, 'peak': res['corr']}

    def evaluate(self, now):
        r = self.measure(now)
        if not r: return []
        self.last = {'lagMs': r['lagMs'], 'peak': r['peak'], 't': now}
        self.lag_history.append(r['lagMs'])
        if len(self.lag_history) > 8:
            self.lag_history.pop(0)

        out = []
        val_range = max(self.lag_history) - min(self.lag_history) if len(self.lag_history) >= 5 else 0
        
        # Humans notice audio leading vision much earlier (~45ms) than audio lagging (~150ms)
        is_leading = r['lagMs'] < -45
        is_lagging = r['lagMs'] > self.drift_ms
        
        offset = (is_leading or is_lagging) and r['peak'] >= 0.25
        drifting = val_range >= self.drift_ms and r['peak'] >= 0.2

        out.append({
            'id': 'LIP_DRIFT',
            'fired': offset or drifting,
            'severity': clamp(max(abs(r['lagMs']), val_range) / 400.0, 0.4, 1.0) if (offset or drifting) else 0.0,
            'detail': f"Audio {'lags' if r['lagMs'] >= 0 else 'leads'} the mouth by {abs(r['lagMs']):.0f} ms (peak correlation {r['peak']:.2f}{f', lag wandered {val_range:.0f} ms' if drifting else ''})."
        })
        out.append({
            'id': 'LIP_DECOUPLED',
            'fired': r['peak'] < 0.12,
            'severity': 0.5 if r['peak'] < 0.12 else 0.0,
            'detail': f"Lip movement and voice barely correlate ({r['peak']:.2f})." if r['peak'] < 0.12 else f"Lips and voice correlate ({r['peak']:.2f})."
        })
        return out

    def probe(self, now):
        r = self.measure(now)
        span = self.window_ms if min(len(self.mouth), len(self.audio)) else 0
        return [
            {
                'key': 'sync', 'label': 'Lip-sync offset', 'progress': 1 if r else clamp(span / self.window_ms, 0, 1) * (0.6 if len(self.mouth) and len(self.audio) else 0),
                'text': f"{'audio behind' if r['lagMs'] >= 0 else 'audio ahead'} {abs(r['lagMs']):.0f} ms" if r else "needs a speaking face and audible voice",
                'bar': clamp(abs(r['lagMs']) / 400.0, 0, 1) if r else 0
            },
            {
                'key': 'corr', 'label': 'Lip/voice correlation', 'progress': 1 if r else 0,
                'text': f"{r['peak']:.2f}" if r else "measuring",
                'bar': clamp(1 - r['peak'] / 0.5, 0, 1) if r else 0
            }
        ]

# --- CompoundDetector ---
class CompoundDetector:
    def __init__(self, window_ms=10000, min_signals=2, cooldown_ms=6000):
        self.window_ms = window_ms
        self.min_signals = min_signals
        self.cooldown_ms = cooldown_ms
        self.live = {}
        self.standing = {}
        self.last_emit = {'t': -float('inf'), 'key': ''}

    def reset(self):
        self.live.clear()
        self.standing.clear()
        self.last_emit = {'t': -float('inf'), 'key': ''}

    def report(self, sid, fired, severity, t):
        if sid not in SIGNALS: return
        def_sig = SIGNALS[sid]
        if def_sig.get('standing'):
            if fired: self.standing[sid] = {'severity': severity}
            else: self.standing.pop(sid, None)
            return
        if fired: self.live[sid] = {'t': t, 'severity': severity}
        else: self.live.pop(sid, None)

    def evaluate(self, now):
        to_delete = [k for k, v in self.live.items() if now - v['t'] > self.window_ms]
        for k in to_delete: del self.live[k]

        ids = sorted(self.live.keys())
        if len(ids) < self.min_signals: return None

        layers = list(set(SIGNALS[i]['layer'] for i in ids))
        standing_ids = sorted(self.standing.keys())
        synergy = [f"{a} + {b}" for a, b in SYNERGY if a in self.live and b in self.live]

        miss = 1.0
        for i in ids: miss *= 1.0 - SIGNALS[i]['weight'] * self.live[i]['severity']
        for i in standing_ids: miss *= 1.0 - 0.5 * SIGNALS[i]['weight'] * self.standing[i]['severity']
        for _ in synergy: miss *= 0.85
        strength = min(1.0, 1.0 - miss)

        key = '+'.join(ids) + '|' + '+'.join(standing_ids)
        is_new = key != self.last_emit['key']
        if not is_new and now - self.last_emit['t'] < self.cooldown_ms: return None
        if is_new and now - self.last_emit['t'] < 1500: return None
        self.last_emit = {'t': now, 'key': key}

        return {
            't': now,
            'signals': ids,
            'layers': layers,
            'standing': standing_ids,
            'synergy': synergy,
            'strength': strength
        }
