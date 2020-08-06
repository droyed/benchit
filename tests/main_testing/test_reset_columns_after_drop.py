import benchit
import numpy as np
funcs = [np.sum,np.prod,np.max,np.mean,np.median]
inputs = [np.random.rand(i,i) for i in 4**np.arange(7)]
t = benchit.timings(funcs, inputs)
print('Starting t :')
print(t)

t.rank()
t.drop(['sum', 'prod'])
print('After ranking and dropping funcs sum and prod t :')
print(t)

t.reset_columns()
print('After resetting columns t :')
print(t)
