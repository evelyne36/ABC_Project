"""Small local tabular discovery and locked-holdout preexperiment; no formal outer test."""
from pathlib import Path
import os,argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-project',type=Path,required=True)
p.add_argument('--xgboost-python-package',type=Path,required=True)
p.add_argument('--lightgbm-package',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); D=a.data_project.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(O/'mpl-cache')
import sys,json,time,hashlib,csv,collections,statistics,importlib.util,warnings,traceback
import numpy as np,pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(a.xgboost_python_package.resolve()));import xgboost as xgb
assert Path(xgb.__file__).resolve().is_relative_to(a.xgboost_python_package.resolve())
lp=a.lightgbm_package.resolve();spec=importlib.util.spec_from_file_location('lightgbm',lp/'__init__.py',submodule_search_locations=[str(lp)]);lgb=importlib.util.module_from_spec(spec);sys.modules['lightgbm']=lgb;spec.loader.exec_module(lgb)
METHODS=['XGB','CrossGB','LGB_constant','LGB_linear']
CONFIGS=[dict(depth=3,ridge=1.,p=1.),dict(depth=3,ridge=10.,p=.5),dict(depth=4,ridge=1.,p=.25)]
SEEDS=[78101,78102]
def hidx(v):return hashlib.sha256(np.asarray(v,dtype=np.int64).tobytes()).hexdigest()
def emit(name,d):
 with (O/name).open('a') as f:f.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n')
def writecsv(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(path,task):
 df=pd.read_csv(path,low_memory=False);y=df.iloc[:,-1];X=df.iloc[:,:-1].copy()
 keep=y.notna()
 if task=='regression':y=pd.to_numeric(y,errors='coerce');keep &= np.isfinite(y)
 X=X.loc[keep].reset_index(drop=True);y=y.loc[keep].reset_index(drop=True)
 X=X.drop(columns=[c for c in X if str(c).strip().lower() in ['predefined.set','unnamed: 0']],errors='ignore')
 for c in X.select_dtypes(include=['object','string']).columns:X[c]=X[c].astype('string').fillna('MISSING').astype(str)
 X=X.replace([np.inf,-np.inf],np.nan)
 return X,y

def split_data(X,y,task):
 if task=='regression':outer,nextest=next(KFold(5,shuffle=True,random_state=42).split(X))
 else:
  cnt=y.value_counts()
  if cnt.min()<10:raise ValueError('minority_class_count_below_10')
  if (task=='binary' and len(cnt)!=2) or len(cnt)>12:raise ValueError('unsupported_class_count')
  outer,nextest=next(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
 disc,audit=train_test_split(outer,test_size=.25,random_state=78000,stratify=None if task=='regression' else y.iloc[outer])
 return np.asarray(outer),np.asarray(nextest),np.asarray(disc),np.asarray(audit)

def prep(X,y,task,it,iv,mode):
 num=list(X.select_dtypes(include=[np.number]).columns);cat=[c for c in X if c not in num]
 trans=[('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num)]
 if mode=='onehot' and cat:trans.append(('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat))
 if mode=='drop' and not num:raise ValueError('no_numeric_features_after_drop')
 pre=ColumnTransformer(trans,remainder='drop')
 A=np.asarray(pre.fit_transform(X.iloc[it]),dtype=np.float32);B=np.asarray(pre.transform(X.iloc[iv]),dtype=np.float32)
 if A.shape[1]>128:raise ValueError('encoded_features_above_128')
 if A.shape[1]<1:raise ValueError('empty_features')
 if task=='regression':
  ym=float(y.iloc[it].mean());ys=max(float(y.iloc[it].std(ddof=0)),1e-12);yt=(y.iloc[it].to_numpy()-ym)/ys;yv=y.iloc[iv].to_numpy();labels=None
 else:
  le=LabelEncoder().fit(y.iloc[it].astype(str));yt=le.transform(y.iloc[it].astype(str));yv=le.transform(y.iloc[iv].astype(str));ym=0.;ys=1.;labels=list(le.classes_)
 return A,B,yt,yv,ym,ys,labels

def make(method,task,cf,seed,trees=100,nclasses=0):
 if method in ['XGB','CrossGB']:
  kw=dict(n_estimators=trees,max_depth=cf['depth'],reg_lambda=cf['ridge'],max_bin=63,learning_rate=.1,tree_method='hist',n_jobs=1,random_state=seed,verbosity=0,subsample=1.,colsample_bytree=1.)
  if method=='CrossGB':kw.update(enable_metagb=True,metagb_pred_colsample_bytree=cf['p'],metagb_lambda_l2=cf['ridge'],metagb_score_agg=0,metagb_min_data_in_leaf=1,metagb_cond_pred_share=False,metagb_leaf_only_use_bias=False,metagb_hist_subtract=False)
  if task=='regression':return xgb.XGBRegressor(**kw)
  return xgb.XGBClassifier(**kw)
 kw=dict(n_estimators=trees,max_depth=cf['depth'],num_leaves=2**cf['depth'],reg_lambda=cf['ridge'],linear_lambda=cf['ridge'],min_child_samples=10,max_bin=63,learning_rate=.1,n_jobs=1,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True,linear_tree=method=='LGB_linear')
 return lgb.LGBMRegressor(**kw) if task=='regression' else lgb.LGBMClassifier(**kw)

def fit_one(method,task,cf,seed,A,B,yt,yv,ym,ys,trees=100):
 model=make(method,task,cf,seed,trees,len(np.unique(yt)));w=time.perf_counter();c=time.process_time()
 with warnings.catch_warnings():warnings.simplefilter('ignore');model.fit(A,yt)
 fw=time.perf_counter()-w;fc=time.process_time()-c;w=time.perf_counter()
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  pred=model.predict(B)*ys+ym if task=='regression' else model.predict_proba(B)
 pw=time.perf_counter()-w
 if not np.isfinite(pred).all():raise ValueError('nonfinite_prediction')
 if task=='regression':val=float(np.sqrt(mean_squared_error(yv,pred)));ll=None
 else:
  ll=float(log_loss(yv,pred,labels=np.arange(pred.shape[1])));val=float(roc_auc_score(yv,pred[:,1])) if task=='binary' else ll
 predictive=None
 if method=='CrossGB':
  conf=model.get_booster().save_config();assert 'grow_metagb_histmaker' in conf or '"enable_metagb":"1"' in conf
  ts=json.loads(model.get_booster().save_raw(raw_format='json'))['learner']['gradient_booster']['model']['trees']
  predictive=sum(0<=t['metagb_leaf_features'][i]<A.shape[1] for t in ts for i,l in enumerate(t['left_children']) if l==-1)
 return dict(primary=val,logloss=ll,fit_wall_sec=fw,fit_cpu_sec=fc,predict_wall_sec=pw,predictive_leaf_count=predictive)

def better(task,x):return -x if task=='binary' else x
# Import historical favourable signals as explicitly retrospective leads.
history=[]
for line in (D/'datacollection0702/experiment_data_reference_0719.md').read_text().splitlines():
 z=[x.strip() for x in line.strip().strip('|').split('|')]
 if len(z)==11 and z[0] in ['random300','random600','xgb_best_mgb30'] and z[2] in ['regression','binary','multiclass']:
  try:
   xv,cv=float(z[5]),float(z[6]);gain=cv-xv if z[4]=='auc' else (xv-cv)/xv
   history.append(dict(protocol=z[0],preprocessing=z[1],task=z[2],dataset=z[3],metric=z[4],xgb=xv,crossgb=cv,gain=gain,reported_winner=z[8]))
  except ValueError:pass
writecsv(O/'historical_leads.csv',history)
files=[];seen=set()
for f in sorted((D/'tabular_extra_random300/data').glob('*.csv')):
 task,name=f.name.split('__',1)
 if name=='friedman1.csv':continue
 files.append((task,name,f))
for f in sorted((D/'dataset/TabArena_regression').glob('*.csv')):files.append(('regression',f.name,f))
for task,folder in [('binary','TabArena_binary'),('multiclass','TabArena_multiclass')]:
 names={x['dataset'] for x in history if x['task']==task and x['gain']>(.0005 if task=='binary' else .001)}
 for f in sorted((D/'dataset'/folder).glob('*.csv')):
  if f.name in names:files.append((task,f.name,f))
unique=[]
for task,name,f in files:
 k=(task,name.lower())
 if k in seen:continue
 seen.add(k);unique.append(dict(task=task,dataset=name,path=str(f.relative_to(D)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
plan=dict(stage='exploratory_discovery_only',inventory=unique,configs=CONFIGS,discovery_seeds=SEEDS,methods=METHODS,n_estimators=100,development_cap=1800,encoded_feature_cap=128,preprocessing='train-fitted median imputation/standardization, categorical onehot or numeric-only drop; modes reported separately; remove predefined.set and Unnamed: 0',split='Outer seed42 first fold held out and never scored; remaining 80% split into discovery75% and audit25% using seed78000. Audit is untouched until shortlist/config freeze.',shortlist_rule='Per task top two cells with CGB better than XGB in both discovery splits, ordered by mean improvement against XGB and LGB_linear; maximum six, fill up to eight with historically-favourable stable cells if available. Modes count as different exploratory cells, not independent datasets.',primary={'regression':'RMSE','binary':'AUC','multiclass':'logloss'},native_library_sha256=hashlib.sha256(Path(xgb.core._LIB._name).resolve().read_bytes()).hexdigest(),xgboost_version=xgb.__version__,lightgbm_version=lgb.__version__,formal_test_evaluations=0,limitations=['legacy local Darwin CrossGB, not controlled Linux build','retrospective dataset leads','no PL/Lin claim from this phase','candidate count matching is not compute matching'])
(O/'frozen_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
rows=[];skips=[];cells=[];start=time.perf_counter()
with threadpool_limits(limits=1):
 for item in unique:
  task,name=item['task'],item['dataset']
  try:
   X,y=load(D/item['path'],task);outer,test,disc,audit=split_data(X,y,task)
   hascat=len(X.select_dtypes(exclude=[np.number]).columns)>0;modes=['onehot','drop'] if hascat else ['numeric']
   if len(X)<100:raise ValueError('fewer_than_100_rows')
   for mode in modes:
    cell=name+'::'+mode;cellrows=[]
    try:
     for seed in SEEDS:
      rng=np.random.default_rng(seed)
      pool=disc
      if len(pool)>1800:
       pool,_=train_test_split(pool,train_size=1800,random_state=seed,stratify=None if task=='regression' else y.iloc[pool])
      it,iv=train_test_split(pool,test_size=.25,random_state=seed,stratify=None if task=='regression' else y.iloc[pool])
      assert not(set(it)|set(iv))&(set(test)|set(audit))
      A,B,yt,yv,ym,ys,labels=prep(X,y,task,it,iv,mode)
      for ci,cf in enumerate(CONFIGS):
       order=list(METHODS);rng.shuffle(order)
       for method in order:
        row=dict(stage='discovery',cell=cell,dataset=name,task=task,mode=mode,seed=seed,config=ci,method=method,n_train=len(it),n_validation=len(iv),n_features=A.shape[1],train_index_sha256=hidx(it),validation_index_sha256=hidx(iv),audit_index_sha256=hidx(audit),outer_test_index_sha256=hidx(test),status='success',error='')
        try:row.update(fit_one(method,task,cf,seed,A,B,yt,yv,ym,ys))
        except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e)[:300],primary=None,logloss=None,fit_wall_sec=None,fit_cpu_sec=None,predict_wall_sec=None,predictive_leaf_count=None)
        rows.append(row);cellrows.append(row);emit('discovery_fits.jsonl',row)
     cells.append(dict(**item,cell=cell,mode=mode,rows=len(X),features=len(X.columns),audit_rows=len(audit)))
     print(task,cell,'done',flush=True)
    except Exception as e:
     entry=dict(cell=cell,dataset=name,task=task,error=type(e).__name__+': '+str(e));skips.append(entry);emit('skipped.jsonl',entry);print('SKIP',cell,entry['error'],flush=True)
  except Exception as e:
   entry=dict(dataset=name,task=task,error=type(e).__name__+': '+str(e));skips.append(entry);emit('skipped.jsonl',entry);print('SKIP',name,entry['error'],flush=True)
summary=[];frozen=[]
for item in cells:
 rs=[r for r in rows if r['cell']==item['cell'] and r['status']=='success'];best={}
 for seed in SEEDS:
  best[seed]={m:min([r for r in rs if r['seed']==seed and r['method']==m],key=lambda r:better(item['task'],r['primary'])) for m in METHODS if any(r['seed']==seed and r['method']==m for r in rs)}
 if not all(all(m in b for m in METHODS) for b in best.values()):continue
 r=dict(cell=item['cell'],dataset=item['dataset'],task=item['task'],mode=item['mode'],rows=item['rows'],features=item['features'],historically_better_than_xgb=any(h['dataset']==item['dataset'] and h['task']==item['task'] and h['gain']>0 for h in history))
 for m in METHODS:
  sel=[b[m] for b in best.values()];r[m+'_primary']=statistics.mean(z['primary'] for z in sel);r[m+'_fit_wall_sec']=statistics.mean(z['fit_wall_sec'] for z in sel)
 for m in ['XGB','LGB_constant','LGB_linear']:
  gains=[b['CrossGB']['primary']-b[m]['primary'] if item['task']=='binary' else (b[m]['primary']-b['CrossGB']['primary'])/max(abs(b[m]['primary']),1e-12) for b in best.values()]
  r['gain_vs_'+m]=statistics.mean(gains);r['wins_vs_'+m]=sum(g>0 for g in gains)
 r['fit_ratio_vs_XGB']=r['CrossGB_fit_wall_sec']/r['XGB_fit_wall_sec'];r['fit_ratio_vs_LGB_linear']=r['CrossGB_fit_wall_sec']/r['LGB_linear_fit_wall_sec']
 for m in METHODS:
  opt=[]
  for ci in range(len(CONFIGS)):
   rr=[z for z in rs if z['method']==m and z['config']==ci]
   if len(rr)==len(SEEDS):opt.append((statistics.mean(z['primary'] for z in rr),ci))
  val,ci=min(opt,key=lambda z:better(item['task'],z[0]));r[m+'_frozen_config']=ci
 summary.append(r)
selected=[]
for task in ['regression','binary','multiclass']:
 eligible=[r for r in summary if r['task']==task and r['wins_vs_XGB']==2]
 # Within each metric family only; no AUC/logloss/RMSE cross-task average.
 eligible.sort(key=lambda r:r['gain_vs_XGB']+.5*r['gain_vs_LGB_linear'],reverse=True)
 chosen_names=set()
 for row in eligible:
  if row['dataset'] in chosen_names:continue
  selected.append(row);chosen_names.add(row['dataset'])
  if len(chosen_names)>=2:break
extras=[r for r in summary if r['historically_better_than_xgb'] and r['wins_vs_XGB']==2 and all(r['dataset']!=z['dataset'] for z in selected)]
extras.sort(key=lambda r:r['gain_vs_XGB'],reverse=True)
selected+=extras[:max(0,8-len(selected))]
# This artifact is written before any audit metric is computed.
freeze={'selection_uses':'discovery metrics only','selected':selected,'configs':CONFIGS,'audit_model_seeds':[79101,79102,79103],'audit_train_cap':4000,'outer_test_evaluations':0}
(O/'shortlist_frozen_before_audit.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
writecsv(O/'discovery_summary.csv',summary)
print('DISCOVERY COMPLETE',len(rows),'fits;',len(summary),'cells; audit shortlist',[(r['dataset'],r['mode']) for r in selected],flush=True)
auditrows=[]
with threadpool_limits(limits=1):
 for selected_row in selected:
  item=next(c for c in cells if c['cell']==selected_row['cell']);task=item['task'];X,y=load(D/item['path'],task);outer,test,disc,audit=split_data(X,y,task)
  for seed in freeze['audit_model_seeds']:
   it=disc
   if len(it)>4000:it,_=train_test_split(it,train_size=4000,random_state=seed,stratify=None if task=='regression' else y.iloc[it])
   assert not set(it)&(set(test)|set(audit))
   A,B,yt,yv,ym,ys,labels=prep(X,y,task,it,audit,item['mode'])
   order=list(METHODS);np.random.default_rng(seed).shuffle(order)
   for m in order:
    ci=selected_row[m+'_frozen_config'];row=dict(stage='locked_audit',cell=item['cell'],dataset=item['dataset'],task=task,mode=item['mode'],seed=seed,method=m,config=ci,n_train=len(it),n_validation=len(audit),n_features=A.shape[1],status='success',error='')
    try:row.update(fit_one(m,task,CONFIGS[ci],seed,A,B,yt,yv,ym,ys))
    except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e),primary=None,logloss=None,fit_wall_sec=None,fit_cpu_sec=None,predict_wall_sec=None,predictive_leaf_count=None)
    auditrows.append(row);emit('audit_fits.jsonl',row)
  print('AUDIT',item['cell'],'done',flush=True)
writecsv(O/'audit_fits.csv',auditrows)
(O/'status.json').write_text(json.dumps(dict(discovery_fits=len(rows),discovery_success=sum(r['status']=='success' for r in rows),cells=len(summary),datasets=len(unique),skipped=skips,audit_fits=len(auditrows),audit_success=sum(r['status']=='success' for r in auditrows),formal_test_evaluations=0,wall_sec=time.perf_counter()-start),ensure_ascii=False,indent=2)+'\n')
print('COMPLETE',round(time.perf_counter()-start,1),'seconds',flush=True)
