import numpy as np
import time

fft_size = 2048
ring = np.zeros(fft_size, dtype=np.float32)
total = 0
filled = 0
since_hop = 0
hop = 640

def push(samples):
    global total, filled, since_hop
    for s in samples:
        ring[total % fft_size] = s
        total += 1
        if filled < fft_size:
            filled += 1
        since_hop += 1
        if since_hop >= hop:
            since_hop = 0
            # frame

samples = np.random.randn(640).astype(np.float32)

t0 = time.time()
for _ in range(5000): # 100 seconds of audio
    push(samples)
print("Time loop:", time.time() - t0)

