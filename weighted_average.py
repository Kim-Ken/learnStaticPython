import numpy as np

l = np.array([1,2,3,4,5,6,7,8,9,10])
w = np.array([0,0,0,0,2,1,0,0,0,0])
print(np.average(l,weights=w)) #ANS 5.3..