# Add benchit path
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg') 
                    
# Import benchit
import benchit
benchit_path = os.path.dirname(os.path.abspath(benchit.__file__))
print('benchit path : '+benchit_path)

# Import existing timed data, convert to benchit benchmarking object and plot
import pandas as pd
df = pd.read_json('minimal_workflow_results.json')
df.index.name = 'Len'
df.columns.name = 'Functions'
t = benchit.bench(df)
t.plot(logx=True)
