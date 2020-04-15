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
import benchit
t = benchit.timings(funcs=[np_noreplace, randsample_on_list, randsample_on_array])
print(t)
