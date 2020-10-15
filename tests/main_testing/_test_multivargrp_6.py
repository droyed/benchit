import numpy as np
import benchit
import matplotlib.pyplot as plt

def new_array(a1, a2):
    a1 = a1 + a2

def inplace(a1, a2):
    a1 += a2

R = np.random.rand

all_inputs = {}

inputs = {str((i,i)):(R(i,i),R(i,i)) for i in 2**np.arange(3,10)}
all_inputs['str_of_tuple'] = inputs

inputs = {(i,i):(R(i,i),R(i,i)) for i in 2**np.arange(3,10)}
all_inputs['tuple'] = inputs

inputs = {i:(R(i,i),R(i,i)) for i in 2**np.arange(3,10)}
all_inputs['scalar'] = inputs

inputs = {(i):(R(i,i),R(i,i)) for i in 2**np.arange(3,10)}
all_inputs['tuple_of_scalar'] = inputs

inputs = {(i,j):(R(i,j),R(i,j)) for i in 2**np.arange(3,10)  for j in 2**np.arange(3,10)}
all_inputs['tuple_combinations'] = inputs

inputs = [(R(i,j),R(i,j)) for i in 2**np.arange(3,10)  for j in 2**np.arange(3,10)]
all_inputs['list'] = inputs

input_names = [None, 'Array-shape', ['arg0len','arg1len']]

#input_names = [None]

for (k,v) in all_inputs.items():    
    for input_name in input_names:
        print('===================================================================')
        print('Processing : '+k)
        t = benchit.timings([new_array,inplace], v, multivar=True, input_name=input_name)
        t.plot(rot=90, save=k+'_'+'input_name : '+str(input_name)+'.png')
        plt.close('all')
