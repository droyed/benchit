import matplotlib.pyplot as plt

import benchit
import numpy as np
funcs = [np.sum,np.prod,np.max,np.mean,np.median]
inputs = [np.random.rand(i,i) for i in 4**np.arange(7)]
t = benchit.timings(funcs, inputs)
t.plot(logy=True, logx=True, save='timings.png')
plt.close("all")
print(t)

tc = t.copy()

s = t.speedups(ref=1) # prod's ref index in t is 1
s.plot(logy=False, logx=True, save='speedups_by_prod.png')
plt.close("all")

st = t.scaled_timings(1) # prod's ref index in t is 1
st.plot(logy=False, logx=True, save='scaledtimings_by_prod.png')
plt.close("all")

t.rank()
t.plot(logy=True, logx=True, save='timings_ranked.png')
t.reset_columns()

t.drop(['sum', 'prod'], axis=1)
t.plot(logy=True, logx=True, save='timings_dropfuncs.png')
plt.close("all")

t = tc.copy()
t.drop([1,16], axis=0)
t.plot(logy=True, logx=True, save='timings_dropdata.png')
plt.close("all")

t = tc.copy()
df = t.to_dataframe()
t = benchit.bench(df, dtype=t.dtype) # or dtype='t'
benchit.bench(t.to_dataframe().iloc[2:],dtype=t.dtype)
benchit.bench(t.to_dataframe().iloc[2:]).plot(logx=True, save='timings_cropdata.png')
plt.close("all")

df['sum+amax'] = df['sum'] + df['amax']
benchit.bench(df).plot(logx=True, save='timings_comb.png')
plt.close("all")
