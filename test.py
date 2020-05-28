import numpy as np

x=[[1,2,3],[1,3,5],[1,2,3]]
y=np.sum(x,axis=0)
print(y)
print(np.argmax(y))