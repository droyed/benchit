import matplotlib.pyplot as plt

import benchit
from pprint import pprint
import numpy as np

#----------------------------- Multivar - Adding arrays -----------------------
def new_array(a1, a2):
    a1 = a1 + a2

def inplace(a1, a2):
    a1 += a2
    
R = np.random.rand
import sys
from collections import OrderedDict
min_ver = float(str(sys.version_info[0])+'.'+str(sys.version_info[1]))

if min_ver<3.6:
    inputs = OrderedDict()
    for i in 2**np.arange(3,13):
        inputs[str((i,i))] = (R(i,i),R(i,i))
else:
    inputs = {str((i,i)):(R(i,i),R(i,i)) for i in 2**np.arange(3,13)}    
    
t = benchit.timings([new_array,inplace], inputs, multivar=True, input_name='Array-shape')
t.plot(logy=True, logx=False, save='multivar_addarrays_timings.png')
plt.close("all")

#--------------------------- Multivar - Euclidean distance --------------------
# Setup input functions
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
fns = [pairwise_distances, cdist]

# Setup input datasets
import numpy as np
if min_ver<3.6:
    in_ = OrderedDict()
    for n in [10,100,500,1000,4000]:
        in_[n] = [np.random.rand(n,3), np.random.rand(n,3)]
else:
    in_ = {n:[np.random.rand(n,3), np.random.rand(n,3)] for n in [10,100,500,1000,4000]}

# Get benchmarking object (dataframe-like) and plot results
t = benchit.timings(fns, in_, multivar=True, input_name='Array-length')
t.plot(logx=True, save='multivar_euclidean_timings.png')
plt.close("all")

#--------------------------- Multivar - groupings -----------------------------
# Setup input datasets
R = np.random.rand
in_ = {(n,W):[R(n,W), R(n,W)] for n in [10, 100, 500, 1000] for W in [3, 5, 8, 10, 20, 50, 80, 100]}

# Get benchmarking object (dataframe-like) and plot results
t = benchit.timings(fns, in_, multivar=True, input_name=['nrows', 'ncols'])
t.plot(logx=True, sp_ncols=2, sp_argID=0, sp_sharey='g', save='multigrp_id0_euclidean_timings.png')
t.plot(logx=True, sp_ncols=2, sp_argID=1, sp_sharey='g', save='multigrp_id1_euclidean_timings.png')
plt.close("all")

#--------------------------- Singlevar - groupings ---------------------------
# Functions
def repeat(b):
    idx = np.flatnonzero(np.r_[b,True])
    return np.repeat(idx[:-1], np.diff(idx))

def maxaccum(b):
    return np.maximum.accumulate(np.where(b,np.arange(len(b)), 0))
    
in_ = {(n,sf): np.random.rand(n)<(100-sf)/100. for n in [100,1000,10000,100000,1000000] for sf in [20, 40, 60, 80, 90, 95]}
t = benchit.timings([repeat, maxaccum], in_, input_name=['Array-length','Sparseness %'])
t.plot(logx=True, sp_ncols=2, save='singlegrp_id0_ffillmask_timings.png')
plt.close("all")

#-------------------------------- Novar ---------------------------------------
# Global inputs
import numpy as np
ar = np.arange(1000000)
l = ar.tolist()
sample_num = 1000

# Setup input functions with no argument
# NumPy random choice on array data
def np_noreplace():
    return np.random.choice(ar, sample_num, replace=False)

from random import sample

# Random sample on list data
def randsample_on_list():
    return sample(l, sample_num)

# Random sample on array data
def randsample_on_array():
    return sample(ar.tolist(), sample_num)

# Benchmark
t = benchit.timings(funcs=[np_noreplace, randsample_on_list, randsample_on_array])
print('')
pprint(t)
