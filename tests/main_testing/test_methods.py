import matplotlib.pyplot as plt

import numpy as np
import benchit

print("--> Benchmarking with funcs as list")
funcs = [np.sum,np.prod,np.max]
inputs = [np.random.rand(i,i) for i in 4**np.arange(7)]

t = benchit.timings(funcs, inputs)
print("--> Printing timings object")
print(t)

print("--> Benchmarking with funcs as dict")
funcs = {'Sum':np.sum,'Prod':np.prod,'Max':np.max}
t = benchit.timings(funcs, inputs)
print("--> Printing timings object")
print(t)

print("--> Printing specs")
benchit.print_specs()

print("--> Printing specs with modules imported from globals")
benchit.print_specs(benchit.extract_modules_from_globals(globals()))

print("--> Plotting timings plot with specs as title & saving it")
saveplot_fname = 'test_methods_output_plot.png'
t.plot(logx=True, save=saveplot_fname)
plt.close("all")

print("--> Plotting timings plot with specs as title & modules in specs & saving it")
saveplot_fname = 'test_methods_output_plot_with_modules.png'
t.plot(logx=True, modules=benchit.extract_modules_from_globals(globals()), save=saveplot_fname)
plt.close("all")
