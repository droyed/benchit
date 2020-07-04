def switch_backend(ID):
    import matplotlib.pyplot as plt
    import matplotlib
    import sys
    
    backends = matplotlib.rcsetup.all_backends
    bkend = backends[ID]
    print('bkend : '+bkend)

    done = True
    try:
        plt.switch_backend(bkend) 
        print('Switching backend to ' + bkend + ' worked!')
    except ImportError:
        print('************* Switching backend to ' + bkend + ' failed!')
        sys.exit()
        done = False
    return done
        

import sys
ID = int(sys.argv[1])
done = switch_backend(ID)

if done==0:
    sys.exit()
    
#==============================================================================    
import numpy as np
funcs = [np.sum,np.prod,np.max]
inputs = [np.random.rand(i) for i in 10**np.arange(5)]

import benchit
t = benchit.timings(funcs, inputs)
t.plot(logy=True, logx=True, save='timings_with_backendID_'+str(ID)+'.png', debug_plotfs=True)
