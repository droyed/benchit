import matplotlib.pyplot as plt

import numpy as np
funcs = [np.sum,np.prod,np.max]
inputs = [np.random.rand(i) for i in 10**np.arange(5)]

import benchit
t = benchit.timings(funcs, inputs)
t.plot(logy=True, logx=True, save='readme_1_timings.png')
plt.close("all")

# Setup input functions
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
fns = [cdist, pairwise_distances]

# Setup input datasets
import numpy as np

import sys
from collections import OrderedDict

min_ver = float(str(sys.version_info[0])+'.'+str(sys.version_info[1]))
if min_ver<3.6:
    in_ = OrderedDict()
    for n in [10,100,500,1000,4000]:
        in_[n] = [np.random.rand(n,3), np.random.rand(n,3)]
else:
    in_ = {n:[np.random.rand(n,3), np.random.rand(n,3)] for n in [10,100,500,1000,4000]}

# Get benchmarking object (dataframe-like) and plot results
t = benchit.timings(fns, in_, multivar=True, input_name='Array-length')
t.plot(logx=True, save='readme_2_timings.png')
plt.close("all")
