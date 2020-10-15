# Setup input functions and datasets
import numpy as np
import benchit

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
          (400,550,'minkowski'):(R(400,3),R(450,3),'minkowski'),
          (200,350,'cityblock'):(R(200,3),R(250,3),'cityblock'),
          (600,650,'cosine'):(R(600,3),R(650,3),'cosine'),
          (200,450,'cityblock'):(R(200,3),R(250,3),'cityblock'),
          (800,850,'euclidean'):(R(800,3),R(800,3),'euclidean')}

# Benchmark
t = benchit.timings(funcs, inputs, multivar=True)
t.plot()

# inputs = {(100,150,'euclidean'):(R(100,3),R(150,3),'euclidean'),
#           (200,150,'cityblock'):(R(200,3),R(150,3),'cityblock'),
          
#           (100,250,'euclidean'):(R(100,3),R(50,3),'euclidean'),
#           (200,250,'cityblock'):(R(200,3),R(250,3),'cityblock'),

#           (100,350,'euclidean'):(R(100,3),R(350,3),'euclidean'),
#           (200,350,'cityblock'):(R(200,3),R(350,3),'cityblock'),

#           (100,450,'euclidean'):(R(100,3),R(450,3),'euclidean'),
#           (200,450,'cityblock'):(R(200,3),R(450,3),'cityblock')}


# # Benchmark
# t = benchit.timings(funcs, inputs, multivar=True)
# t.plot(sp_argID=0)
# t.plot(sp_argID=1, sp_ncols=1)
# t.plot(sp_argID=2)
