import benchit
import numpy as np
funcs = [np.sum,np.prod,np.max,np.mean,np.median]
inputs = [np.random.rand(i,i) for i in 4**np.arange(7)]
t = benchit.timings(funcs, inputs)

t.speedups(0).plot(logx=True, save='test_speedups_with_ref_as_int.png')
t.speedups('sum').plot(logx=True, save='test_speedups_with_ref_as_str.png')
t.speedups(np.sum).plot(logx=True, save='test_speedups_with_ref_as_func.png')

t.scaled_timings(0).plot(logx=True, save='test_scaledtimings_with_ref_as_int.png')
t.scaled_timings('sum').plot(logx=True, save='test_scaledtimings_with_ref_as_str.png')
t.scaled_timings(np.sum).plot(logx=True, save='test_scaledtimings_with_ref_as_func.png')
