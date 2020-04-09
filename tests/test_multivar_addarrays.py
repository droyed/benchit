# Setup input functions and datasets
def func1(a1, a2): 
    a1 = a1 + a2

def func2(a1, a2):
    a1 += a2

funcs = [func1, func2]

import numpy as np
inputs = {str((i,i)):(np.random.rand(i,i),np.random.rand(i,i)) for i in 2**np.arange(3,13)}

# Benchmark
import benchit
t = benchit.timings([func1,func2], inputs, multivar=True, input_name='Array-shape')
t.plot(savepath='multivar_addarrays_timings.png')
