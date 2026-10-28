from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from baselines.estimators import GBDTPLNative
X=np.random.default_rng(42).normal(size=(112,3));y=X[:,0]+X[:,1]
print('create',flush=True)
m=GBDTPLNative({'num_trees':3,'verbose':1,'num_leaves':4,'max_var':3,'num_bins':31,'min_sum_hessians':4})
print('fit',flush=True)
m.fit(X,y)
print('predict',flush=True)
print(m.predict(X[:3]),flush=True)
