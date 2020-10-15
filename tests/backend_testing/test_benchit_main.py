## Syntax to run as a bash script :
# $ for i in {0..30}; do python3 test_benchit_main.py $i; python test_benchit_main.py $i; done


import sys
import matplotlib.pyplot as plt

if len(sys.argv)>1:
    ID = int(sys.argv[1])    
    import matplotlib
    backends = matplotlib.rcsetup.all_backends
    bkend = backends[ID]
    try:
        plt.switch_backend(bkend) 
        print('Switching backend to ' + bkend + ' worked!')
    except ImportError:
        print('************* Switching backend to ' + bkend + ' failed!')
        sys.exit()
else:
    print('No matplotlib backend ID provided. Hence, going with the default one.')


# ----------------------------------------------------------------------------
# Add benchit path
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import benchit

## Un-comment for notebook runs and set backend manually
#%matplotlib inline

## Un-comment for notebook specific and non-interactive backend runs
#benchit.setparams(environ='notebook')

import matplotlib

bkend = matplotlib.get_backend()
print('Backend set finally : '+bkend)

from utils import save_all_params_figures, get_sample_timings_obj_from_json, setup_outdir

t = benchit.bench(get_sample_timings_obj_from_json('minimal_workflow_results.json'))
outdir = setup_outdir(bkend)
save_all_params_figures(t, outdir)
