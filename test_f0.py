import numpy as np
from server.analysis_metrics import estimate_f0
import time

buf = np.random.randn(2048).astype(np.float32)

t0 = time.time()
for _ in range(1000):
    estimate_f0(buf, 32000)
print("Time old:", time.time() - t0)

