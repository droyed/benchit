# Setup input functions and datasets
import numpy as np
funcs = [np.sum,np.prod,np.max]
inputs = [np.random.rand(i) for i in 10**np.arange(5)]

# Benchmark
import benchit
t = benchit.timings(funcs, inputs)
t.plot(logy=True, logx=True, savepath='singlevar_numpy_ufuncs_timings.png')

s = t.speedups(ref_func_by_index=1) # prod's index in t is 1
s.plot(logy=False, logx=True, savepath='singlevar_numpy_ufuncs_speedups_by_prod.png')
