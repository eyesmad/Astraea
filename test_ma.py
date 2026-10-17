import numpy as np
from server.analysis_metrics import moving_average, resample

a = np.array([1,2,3,4,5], dtype=float)
print("MA2:", moving_average(a, 2))
print("MA3:", moving_average(a, 3))

# JS:
# sum += arr[i]
# if (i >= n) sum -= arr[i-n]
# out[i] = sum / min(n, i+1)

