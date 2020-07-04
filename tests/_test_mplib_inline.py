import benchit
import numpy as np
funcs = [np.sum,np.prod,np.max]
inputs = [np.random.rand(i) for i in 10**np.arange(5)]
t = benchit.timings(funcs, inputs)

%matplotlib inline
t.plot(logx=True, save='inline_plot.png')
t.plot(logx=True, figsize=(20,9), specs_fontsize=12, save='inline_plot_with_adjusted_params.png')
