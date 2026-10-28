"""Shared local exploratory model interface, copied from the executed scout."""
from pathlib import Path
import os,sys,json,time,importlib.util,warnings
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits

def initialize(xgb_package,lgb_package,cache):
    global xgb,lgb
    os.environ['MPLCONFIGDIR']=str(cache)
    sys.path.insert(0,str(Path(xgb_package).resolve()))
    import xgboost as xgb
    assert Path(xgb.__file__).resolve().is_relative_to(Path(xgb_package).resolve())
    lp=Path(lgb_package).resolve()
    spec=importlib.util.spec_from_file_location('lightgbm',lp/'__init__.py',submodule_search_locations=[str(lp)])
    lgb=importlib.util.module_from_spec(spec);sys.modules['lightgbm']=lgb;spec.loader.exec_module(lgb)

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

