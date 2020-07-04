import numpy as np

def numpy_concat(a):
    return np.concatenate([a, a])

funcs = {'r_':lambda a:np.r_[a, a],
         'stack+reshape':lambda a:np.stack([a, a]).reshape(-1),
         'hstack':lambda a:np.hstack([a, a]),
         'concat':numpy_concat,
         'tile':lambda a:np.tile(a,2)}

inputs = [np.random.rand(2**k) for k in range(15)]

import benchit
t = benchit.timings(funcs, inputs, multivar=False)
t.rank()
t.plot(logx=True, save='timings_lambdasmix.png')
