import matplotlib.pyplot as plt

# Setup input functions and datasets
import numpy as np
funcs = [np.sum,np.prod,np.max]
inputs = [np.random.rand(i) for i in 10**np.arange(5)]

# Benchmark and plot
import benchit
t = benchit.timings(funcs, inputs)
t.plot(logy=True, logx=True, save='index_timings.png')
plt.close("all")
