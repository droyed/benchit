# Setup input functions and datasets
import numpy as np
funcs = [np.sum,np.prod,np.max]
inputs = {(i,ax):(np.random.rand(i,i,i),ax) for i in [10,20,100,200,300,400,500] for ax in [0,1,2]}

# Benchmark and plot
import benchit
t = benchit.timings(funcs, inputs, multivar=True, input_name=['Array lengths along each axis', 'Axis'])
t.plot(logy=True, logx=False, sp_argID=1, sp_ncols=2)
