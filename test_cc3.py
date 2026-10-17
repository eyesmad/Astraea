import numpy as np
from server.analysis_metrics import cross_correlate, zscore

a = np.array([0,0,1,2,3,4,5,4,3])
b = np.array([1,2,3,4,5,4,3,0,0]) # b is ahead of a by 2

print(cross_correlate(zscore(a), zscore(b), 5))

