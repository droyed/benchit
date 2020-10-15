# Setup input functions and datasets
import numpy as np

from scipy.spatial.distance import cdist,pdist,squareform

def pdist_based(a, b, m):
    return squareform(pdist(a, metric=m))

cdist2 = cdist
cdist3 = cdist
cdist4 = cdist
cdist5 = cdist

funcs = [cdist,pdist_based,cdist2,cdist3,cdist4,cdist5]

R = np.random.rand
inputs = {(100,150,'euclidean'):(R(100,3),R(150,3),'euclidean'),
          (200,250,'cityblock'):(R(200,3),R(250,3),'cityblock'),
          (400,450,'minkowski'):(R(400,3),R(450,3),'minkowski'),
          (600,650,'cosine'):(R(600,3),R(650,3),'cosine'),
          (800,850,'euclidean'):(R(800,3),R(800,3),'euclidean')}

# Benchmark
import benchit
t = benchit.timings(funcs, inputs, multivar=True)
t.plot(logx=True)
