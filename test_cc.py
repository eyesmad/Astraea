import numpy as np
from server.analysis_metrics import cross_correlate, zscore, moving_average, resample

a = np.array([1,2,3,4,5,4,3,2,1])
b = np.array([0,0,1,2,3,4,5,4,3]) # delayed by 2

print(cross_correlate(zscore(a), zscore(b), 5))

