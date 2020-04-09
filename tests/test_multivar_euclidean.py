# Setup input functions
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
fns = [cdist, pairwise_distances]

## Use custom names as keys in a dict 
#fns = {'Scipy cdist':cdist, 'Scikit-learn pairwise_distances':pairwise_distances}

# Setup input datasets
import numpy as np
in_ = {(n,3):[np.random.rand(n,3), np.random.rand(n,3)] for n in [10,100,500,1000,4000]}

# Benchmark
import benchit
t = benchit.timings(fns, in_, multivar=True)
t.plot(savepath='multivar_euclidean_timings.png')
