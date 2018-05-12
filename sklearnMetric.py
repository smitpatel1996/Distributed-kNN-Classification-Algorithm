import dask_ml.joblib
import time
from dask.distributed import Client
from sklearn.externals.joblib import parallel_backend
from sklearn.datasets import load_iris
from sklearn.neighbors import BallTree
import pandas as pd
import numpy as np
a = pd.read_csv('TrainSet.csv')
target = a[[' UNS']].values
data = a.as_matrix(columns=a.columns[0:5])
b = pd.read_csv('TestSet.csv')
testtarget = b[[' UNS']].values
testdata = b.as_matrix(columns=b.columns[0:5])
client = Client()
strt = time.time()
model = BallTree(data,leaf_size=5)
with parallel_backend('dask', scatter=[data, target]):
	c = 0
	for i in range(0,50):
		dist,ind = model.query([testdata[i]],k=5)
		l = list(map(lambda x: target[x][0],ind[0]))	
		o = max(set(l), key=l.count)
		if(o == testtarget[i][0]):
			c = c+1
	print c			
end = time.time()
tm = end-strt
print round(tm,3)