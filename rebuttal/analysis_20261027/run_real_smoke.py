"""Exploratory training-only dataset triage; never scores reserved outer-test rows."""
from pathlib import Path
import os
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-project',type=Path,required=True,help='Legacy project containing tabular_extra_random300/data and dataset/')
parser.add_argument('--xgboost-python-package',type=Path,required=True,help='Explicit custom CrossGB Python package with a working native library')
parser.add_argument('--lightgbm-package',type=Path,help='Optional existing lightgbm package directory; otherwise use the installed distribution')
parser.add_argument('--output',type=Path,required=True,help='New directory; never overwrite a discovery run')
args=parser.parse_args()
BASE=Path(__file__).resolve().parents[1];DATA_ROOT=args.data_project.resolve();OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=False)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(OUT/'mpl-cache')
import sys,json,hashlib,time,csv,statistics,importlib.util,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import KFold,train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(args.xgboost_python_package.resolve()))
import xgboost as xgb
assert Path(xgb.__file__).resolve().is_relative_to(args.xgboost_python_package.resolve()), 'Unexpected xgboost package'
if args.lightgbm_package:
 lp=args.lightgbm_package.resolve()
 spec=importlib.util.spec_from_file_location('lightgbm',lp/'__init__.py',submodule_search_locations=[str(lp)])
 lgb=importlib.util.module_from_spec(spec);sys.modules['lightgbm']=lgb;spec.loader.exec_module(lgb)
else:
 import lightgbm as lgb
TAB=DATA_ROOT/'tabular_extra_random300/data';ARENA=DATA_ROOT/'dataset/TabArena_regression'
files={d:TAB/('regression__'+d+'.csv') for d in ['kin8nm','energy_efficiency','abalone','Moneyball','naval_propulsion_plant','cpu_activity']}
files.update({d:ARENA/(d+'.csv') for d in ['airfoil_self_noise','concrete_compressive_strength','houses','wine_quality']})
configs=[dict(depth=3,ridge=1.0),dict(depth=4,ridge=10.0)]
seeds=[20261027,20261028]
plan={'purpose':'exploratory dataset triage only; not formal benchmark or evidence of population superiority','datasets':{d:str(p.relative_to(DATA_ROOT)) for d,p in files.items()},'seeds':seeds,'configs':configs,'n_estimators':60,'max_bin':63,'outer_test':'KFold(5,shuffle=True,random_state=42), first fold reserved and never scored','development_only':'Each repeat samples <=1500 rows from outer-training pool, then 75/25 train/validation; preprocessing fit on development-train only','methods':['XGB','CrossGB','LGB_constant','LGB_linear'],'native_scope':'Explicit user-selected custom build; provenance recorded, no automatic equivalence claim to the formal Linux library. Results are discovery-only.','xgboost_version':xgb.__version__,'lightgbm_version':lgb.__version__,'native_sha256':hashlib.sha256(Path(xgb.core._LIB._name).resolve().read_bytes()).hexdigest(),'source_hashes':{d:hashlib.sha256(p.read_bytes()).hexdigest() for d,p in files.items()},'fit_budget':160,'formal_test_evaluations':0}
(OUT/'frozen_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
rows=[];start=time.perf_counter()
with threadpool_limits(limits=1):
 for dataset,path in files.items():
  df=pd.read_csv(path);y=pd.to_numeric(df.iloc[:,-1],errors='coerce');valid=np.isfinite(y);df=df.loc[valid].reset_index(drop=True);y=y[valid].to_numpy();X=df.iloc[:,:-1].copy()
  for c in X.select_dtypes(include=['object','string']).columns:X[c]=X[c].astype('string').fillna('MISSING').astype(str)
  nums=list(X.select_dtypes(include=[np.number]).columns);cats=[c for c in X.columns if c not in nums]
  outer_train,outer_test=next(KFold(5,shuffle=True,random_state=42).split(X))
  for seed in seeds:
   rng=np.random.default_rng(seed);pool=rng.choice(outer_train,size=min(1500,len(outer_train)),replace=False)
   it,iv=train_test_split(pool,test_size=.25,random_state=seed)
   assert not (set(it)|set(iv))&set(outer_test)
   pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],remainder='drop')
   A=np.asarray(pre.fit_transform(X.iloc[it]),dtype=np.float32);B=np.asarray(pre.transform(X.iloc[iv]),dtype=np.float32)
   ym,ys=float(np.mean(y[it])),float(np.std(y[it]));ys=max(ys,1e-12);yt=(y[it]-ym)/ys
   for ci,cf in enumerate(configs):
    for method in plan['methods']:
     t=time.perf_counter();c=time.process_time()
     row={'dataset':dataset,'seed':seed,'config':ci,'method':method,'n_train':len(it),'n_validation':len(iv),'n_features':A.shape[1],'status':'success','error':''}
     try:
      if method in ['XGB','CrossGB']:
       p=dict(n_estimators=60,learning_rate=.1,max_depth=cf['depth'],reg_lambda=cf['ridge'],max_bin=63,tree_method='hist',n_jobs=1,random_state=seed,verbosity=0,subsample=1.,colsample_bytree=1.)
       if method=='CrossGB':p.update(enable_metagb=True,metagb_pred_colsample_bytree=1.,metagb_score_agg=0,metagb_lambda_l2=cf['ridge'],metagb_min_data_in_leaf=1,metagb_cond_pred_share=False,metagb_leaf_only_use_bias=False,metagb_hist_subtract=False)
       model=xgb.XGBRegressor(**p)
      else:
       model=lgb.LGBMRegressor(n_estimators=60,learning_rate=.1,max_depth=cf['depth'],num_leaves=2**cf['depth'],reg_lambda=cf['ridge'],linear_lambda=cf['ridge'],min_child_samples=10,max_bin=63,linear_tree=method=='LGB_linear',n_jobs=1,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)
      model.fit(A,yt);row['fit_wall_sec']=time.perf_counter()-t;row['fit_cpu_sec']=time.process_time()-c
      if method=='CrossGB':
       trees=json.loads(model.get_booster().save_raw(raw_format='json'))['learner']['gradient_booster']['model']['trees'];assert all('metagb_leaf_features' in z for z in trees)
      with warnings.catch_warnings():warnings.simplefilter('ignore');pred=model.predict(B)*ys+ym
      if not np.all(np.isfinite(pred)):raise ValueError('nonfinite predictions')
      row['validation_rmse']=float(np.sqrt(np.mean((pred-y[iv])**2)));row['scaled_validation_rmse']=row['validation_rmse']/ys
     except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e),validation_rmse=None,scaled_validation_rmse=None,fit_wall_sec=time.perf_counter()-t,fit_cpu_sec=time.process_time()-c)
     rows.append(row)
     with (OUT/'fits.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(dataset,'completed',flush=True)
summary=[]
for dataset in files:
 best={}
 for seed in seeds:
  best[seed]={}
  for m in plan['methods']:
   good=[r for r in rows if r['dataset']==dataset and r['seed']==seed and r['method']==m and r['status']=='success']
   if good:best[seed][m]=min(good,key=lambda r:r['validation_rmse'])
 r={'dataset':dataset}
 for m in plan['methods']:
  selected=[b[m] for b in best.values() if m in b];r[m+'_validation_rmse']=statistics.mean(z['validation_rmse'] for z in selected) if selected else None;r[m+'_selected_fit_wall_sec']=statistics.mean(z['fit_wall_sec'] for z in selected) if selected else None
 for competitor in ['XGB','LGB_constant','LGB_linear']:
  pairs=[(b['CrossGB']['validation_rmse'],b[competitor]['validation_rmse']) for b in best.values() if 'CrossGB' in b and competitor in b]
  r['CrossGB_improvement_vs_'+competitor]=statistics.mean((v-c)/v for c,v in pairs) if pairs else None;r['CrossGB_wins_vs_'+competitor]=sum(c<v for c,v in pairs)
 summary.append(r)
with (OUT/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(OUT/'status.json').write_text(json.dumps({'fits':len(rows),'success':sum(r['status']=='success' for r in rows),'outer_test_evaluations':0,'wall_sec':time.perf_counter()-start,'limitations':['local legacy build','two tiny configurations','validation reused for method tuning and dataset triage','no PL/LinXGB run','different from archived formal protocols','no claim of independent confirmation']},indent=2)+'\n')
print('COMPLETE',len(rows),'fits;',round(time.perf_counter()-start,1),'sec',flush=True)
