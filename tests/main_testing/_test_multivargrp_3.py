# Setup input functions and datasets
import numpy as np

from scipy.spatial.distance import cdist,pdist,squareform

def pdist_based(a, b, m):
    return squareform(pdist(a, metric=m))

funcs = [cdist,pdist_based]
inputs = {(i,j,metric):(np.random.rand(i,3),np.random.rand(j,3),metric) for i in [10,20,50,100]  for j in [100,200,500,1000] for metric in ['euclidean', 'cityblock']}

inputs = {(i,j,metric):(np.random.rand(i,3),np.random.rand(j,3),metric) for i in [10,20,30]  for j in [100,200] for metric in ['euclidean', 'cityblock', 'minkowski', 'cosine']}

inputs = {(i,j,metric):(np.random.rand(i,3),np.random.rand(j,3),metric) for i in [16, 29, 56]  for j in [134,225] for metric in ['euclidean', 'cityblock', 'minkowski', 'cosine']}

inputs = {(i,j,metric):(np.random.rand(i,3),np.random.rand(j,3),metric) for i in [16, 790, 10900]  for j in [134,2250] for metric in ['euclidean', 'cityblock', 'minkowski', 'cosine']}

# Benchmark
import benchit
t = benchit.timings(funcs, inputs, multivar=True, input_name=['Array1', 'Array2', 'metric'])

t.plot(logx=True, sp_argID=0, sp_ncols=2)
t.plot(logx=True, sp_argID=1, sp_ncols=2)
t.plot(logx=False, sp_argID=2, sp_ncols=2)
