import numpy as np
import time
from server.analysis_metrics import spectrum_db, estimate_f0, brickwall_drop, spectral_flatness

frame = np.random.randn(2048).astype(np.float32)
sr = 32000
n = 2048

t0 = time.time()
for _ in range(5000): # 100 seconds
    freq = spectrum_db(frame)
    hz = sr / n
    flat = spectral_flatness(freq, int(300 / hz), int(4000 / hz))
    f0 = estimate_f0(frame, sr)
    drop = brickwall_drop(freq, sr, n)

print("Time frame:", time.time() - t0)

