"""Frame-level audio features.

These mirror public/js/audioFeatures.js and server/dsp.js exactly: same windows, same thresholds, same
return values. The JS versions stay the reference implementation and are covered by `npm test`; the Python
ones are what the analysis server actually runs, and tests/test_analysis.py checks the two agree.
"""
import numpy as np

_WINDOWS = {}


def _blackman(n):
    if n not in _WINDOWS:
        i = np.arange(n)
        _WINDOWS[n] = 0.42 - 0.5 * np.cos(2 * np.pi * i / n) + 0.08 * np.cos(4 * np.pi * i / n)
    return _WINDOWS[n]


def spectrum_db(frame):
    """n/2 magnitude bins in dB, floored at -120, like AnalyserNode.getFloatFrequencyData."""
    n = len(frame)
    spec = np.fft.rfft(frame * _blackman(n))[: n // 2]
    return np.maximum(-120.0, 20 * np.log10(np.abs(spec) / n + 1e-12))


def rms(buf):
    return float(np.sqrt(np.mean(np.square(buf)))) if len(buf) else 0.0


def spectral_flatness(freq_db, lo_bin, hi_bin):
    """~1 for noise, ~0 for tonal sounds. Geometric over arithmetic mean of power."""
    if hi_bin - lo_bin <= 0:
        return 0.0
    p = np.power(10.0, freq_db[lo_bin:hi_bin] / 10.0) + 1e-12
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def estimate_f0(buf, sample_rate, fmin=70, fmax=400):
    """Normalised autocorrelation pitch estimate. Returns (f0, clarity); f0 is 0 when unvoiced."""
    min_lag = int(sample_rate // fmax)
    max_lag = min(int(sample_rate // fmin), len(buf) - 1)
    n = len(buf) - max_lag
    if n <= 0:
        return 0.0, 0.0
    seg = buf[:n]
    e0 = float(np.dot(seg, seg))
    if e0 < 1e-9:
        return 0.0, 0.0

    # Row `i` is buf[i:i+n], so rows min_lag..max_lag are exactly the lagged copies we need.
    windows = np.lib.stride_tricks.sliding_window_view(buf, n)[min_lag : max_lag + 1]
    s = windows @ seg
    e1 = np.einsum("ij,ij->i", windows, windows)
    r = s / np.sqrt(e0 * e1 + 1e-12)

    thresh = 0.85 * float(np.max(r))
    # The first strong local maximum, which avoids reporting an octave too low.
    for k in range(1, len(r) - 1):
        if r[k] >= thresh and r[k] >= r[k - 1] and r[k] >= r[k + 1]:
            lag = min_lag + k
            return float(sample_rate / lag), float(r[k])
    return 0.0, 0.0


def brickwall_drop(freq_db, sample_rate, fft_size):
    """Abrupt spectral cliff at or above 5 kHz followed by an empty band: a band-limited vocoder or codec.

    Narrowband phone audio cuts below 5 kHz and is deliberately not matched. Returns (drop_db, at_hz).
    """
    bin_hz = sample_rate / fft_size
    band_bins = max(1, int(round(500 / bin_hz)))
    count = len(freq_db) // band_bins
    if count < 4:
        return 0.0, 0.0
    bands = freq_db[: count * band_bins].reshape(count, band_bins).mean(axis=1)
    hz = np.arange(count) * band_bins * bin_hz

    best_drop, best_hz = 0.0, 0.0
    for i in range(1, count - 2):
        if hz[i] < 5000:
            continue
        drop = bands[i - 1] - bands[i]
        tail = max(bands[i + 1], bands[i + 2])
        if drop > best_drop and bands[i - 1] - tail > drop * 0.9:
            best_drop, best_hz = float(drop), float(hz[i])
    return best_drop, best_hz


def decimate2(x):
    """2:1 decimation with a short windowed-sinc low-pass, for feeding 16 kHz models from 32 kHz capture."""
    taps = 31
    i = np.arange(taps)
    m = i - (taps - 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(m == 0, 0.5, np.sin(np.pi * 0.5 * m) / (np.pi * np.where(m == 0, 1, m)))
    h = sinc * (0.54 - 0.46 * np.cos(2 * np.pi * i / (taps - 1)))
    h /= h.sum()
    return np.convolve(x, h, mode="same")[::2].astype(np.float32)


# ---- statistics shared with public/js/stats.js ----

def percentile(a, p):
    if len(a) == 0:
        return 0.0
    s = np.sort(np.asarray(a, dtype=float))
    i = min(max(p, 0.0), 1.0) * (len(s) - 1)
    lo, hi = int(np.floor(i)), int(np.ceil(i))
    return float(s[lo] + (s[hi] - s[lo]) * (i - lo))


def median(a):
    return percentile(a, 0.5)


def robust_std(a):
    """Interquartile range scaled to match the standard deviation of a normal distribution."""
    return (percentile(a, 0.75) - percentile(a, 0.25)) / 1.349


def std(a):
    return float(np.std(np.asarray(a, dtype=float), ddof=1)) if len(a) > 1 else 0.0


def clamp(v, lo, hi):
    return min(hi, max(lo, v))
