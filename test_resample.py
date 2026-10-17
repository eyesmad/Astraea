import numpy as np
from server.analysis_metrics import resample

track = [
  {'t': 0, 'v': 10},
  {'t': 40, 'v': 20},
  {'t': 80, 'v': 30}
]
print(resample(track, 25, 0, 80)) # step = 1000/25 = 40. expect [10, 20, 30]

