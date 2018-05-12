import dask_ml.joblib
import time
from dask.distributed import Client
from sklearn.externals.joblib import parallel_backend
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
a = pd.read_csv('TrainSet.csv')
target = a[[' UNS']].values
data = a.as_matrix(columns=a.columns[0:5])
b = pd.read_csv('TestSet.csv')
testtarget = b[[' UNS']].values
testdata = b.as_matrix(columns=b.columns[0:5])
client = Client()
strt = time.time()
model = KNeighborsClassifier(n_neighbors=5)
with parallel_backend('dask', scatter=[data, target]):
	model.fit(data,target.ravel())
	c = 0
	for i in range(0,100):
		o = model.predict([testdata[i]])
		if(o[0] == testtarget[i][0]):
			c = c+1
	print c		
end = time.time()
tm = end-strt
print round(tm,3)
