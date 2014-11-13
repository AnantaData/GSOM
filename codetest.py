import numpy as np

a = np.array([1,0,1])
b = np.array([0,1,1])

aa= a.astype(bool)
bb=b.astype(bool)

union = np.where(np.logical_or(aa,bb)==False)[0]

print len(union)