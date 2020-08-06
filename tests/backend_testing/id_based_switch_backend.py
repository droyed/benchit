def switch_backend(ID):
    import matplotlib.pyplot as plt
    import matplotlib
    import sys
    
    backends = matplotlib.rcsetup.all_backends
    bkend = backends[ID]
    print('===================================================================')
    print('bkend : '+bkend+'. ID : '+str(ID))

    done = True
    try:
        plt.switch_backend(bkend) 
        print('Switching backend to ' + bkend + ' worked!')
    except ImportError:
        print('************* Switching backend to ' + bkend + ' failed!')
        sys.exit()
        done = False
    return done, bkend
        

import sys
ID = int(sys.argv[1])
done, bkend = switch_backend(ID)

if not done:
    sys.exit()
    
#==============================================================================    
import benchit
import pandas as pd
df = pd.read_json('minimal_workflow_results.json')
df.index.name = 'Len'
df.columns.name = 'Functions'
t = benchit.bench(df)

t.plot(logx=True, save='timings_with_backend=> '+bkend+'_ID_'+str(ID)+'.jpg', debug_plotfs=True)
