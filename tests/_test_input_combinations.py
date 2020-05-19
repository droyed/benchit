from __future__ import print_function
import numpy as np
import pandas as pd
import networkx as nx
import benchit

# For quick view, on Linux systems :
# python test_input_combinations.py > .tmp.txt; cat .tmp.txt
# python3 test_input_combinations.py > .tmp.txt; cat .tmp.txt

def f(i):
    return

def print_inputs(in_):
    if isinstance(in_, dict):
        for i,(in_k,d) in enumerate(zip(in_.keys(), in_.values())):
            print('Input #'+str(i+1)+' :')
            print('Key - '+str(in_k))
            print('Data -')
            print(d)            
    else:
        for i,d in enumerate(in_):
            print('Input #'+str(i+1)+':')
            print('Data -')
            print(d)
    
ar1 = np.random.randint(0,9,(3,4))
ar2 = np.random.randint(0,9,(5,6))
lst1 = list(range(100,105))
lst2 = list(range(1000,1004))

in_ar = [ar1,ar2]
in_df = [pd.DataFrame(ar1), pd.DataFrame(ar2)]
in_l = [lst1,lst2]
in_t = [(3,4),(2,3,8),(1,8,4,2,5,6)]
in_s = lst1
in_m = [ar1,lst1]
in_r = [nx.lollipop_graph(4,6).edges]

all_inputs = {'List':in_l,'Array':in_ar,'Dataframe':in_df,'Tuple':in_t,'Scalar':in_s,'Mixed':in_m,'Other (Networkx-Edgeview)':in_r}

for USE_INPUT_AS_DICT in [False, True]:
    for in_,v in zip(all_inputs.keys(), all_inputs.values()):
        sep = ' ====================== '
                
        in_type = 'dict' if USE_INPUT_AS_DICT==1 else 'list'     
        print("\n"+sep + 'Input class : ' + in_type + " . Each input class : "+in_ + sep)

        if USE_INPUT_AS_DICT==1:
            v = dict(zip(['key'+str(i) for i in range(len(v))], v))

        print_inputs(v)
        
        all_possible_indexbys = benchit._get_possible_indexbys(v)
        print("'auto' Indexby (applicable when not a dict, otherwise it's keys) : " + all_possible_indexbys[0] + ". All other possible indexbys : " + str(all_possible_indexbys))
        print("Output for each indexby value :")
        for o in ['auto','len','shape','size','item','scalar']:
            print(' Option = '+o+'. ', end='')                
            success = True
            try:
                benchit.timings([f],v,indexby=o)
            except ValueError as E:
                success = False
                print("Status = Fail. Value error - "+str(E))
                
            if success==1:
                index = benchit.timings([f],v,indexby=o).get_dataframe().index.tolist()
                print("Status = Success! Index : " + str(index))
