"""Bounded expansion of eight existing local real datasets; fixed prior configurations."""
from pathlib import Path
import argparse,json,csv,time,statistics,hashlib,collections
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-project',type=Path,required=True);p.add_argument('--xgboost-python-package',type=Path,required=True);p.add_argument('--lightgbm-package',type=Path,required=True);p.add_argument('--previous-results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();D=a.data_project.resolve();R=a.previous_results.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
import model_runtime as rt
rt.initialize(a.xgboost_python_package,a.lightgbm_package,O/'mpl-cache');np=rt.np
base=json.loads((R/'optimistic_scout_v1__frozen_plan.json').read_text());previous=list(csv.DictReader((R/'optimistic_scout_v1__discovery_summary.csv').open()))
NAMES=['wine_quality.csv','physiochemical_protein.csv','miami_housing.csv','QSAR_fish_toxicity.csv','healthcare_insurance_expenses.csv','breast-w.csv','maternal_health_risk.csv','website_phishing.csv']
items=[r for r in previous if r['dataset'] in NAMES];METHODS=['XGB','CrossGB','LGB_constant','LGB_linear'];SEEDS=[89101,89102,89103,89104,89105]
for item in items:
 for m in METHODS:item[m+'_frozen_config']=int(item[m+'_frozen_config'])
plan={'datasets':NAMES,'cells':items,'repeat_seeds':SEEDS,'n_estimators':100,'configs':base['configs'],'parameter_selection':'Frozen from previous two discovery splits; no new HPO on these validation labels. All original prior configurations retained.','split':'Preserve seed42 outer1 test unscored; each repeat repartitions the remaining outer-training pool 75/25, stratified for classification. Cap training at4000; validation is not subsampled.','preprocessing':'Same train-fitted imputation/scaling and categorical mode as previous scout; modes are not independent datasets.','promotion_rule':'A: mean gain vs XGB>=1% RMSE/logloss or >=.002 AUC, wins>=4/5; positive mean gain and wins>=3/5 against both LGB variants. B: stable/practically positive vs XGB but not all neighbours. C: remaining positive/weak signals. D: nonpositive vs XGB. Cost recorded separately, no automatic speed claim.','limitations':['exploratory repeated overlapping splits, not 5 independent datasets','retrospective data and previously examined discovery labels; not pristine confirmation','not a formal benchmark; outer-test never scored','legacy local native build and fixed small parameter space','no PL/Lin measured in this expansion'],'native_library_sha256':base['native_library_sha256'],'lightgbm_version':base['lightgbm_version'],'fit_attempts_expected':len(items)*len(SEEDS)*len(METHODS),'no_additional_datasets_after_seeing_results':True}
(O/'frozen_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
rows=[];splits=[];start=time.perf_counter()
def emit(name,r):
 with (O/name).open('a') as f:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def h(v):return hashlib.sha256(np.asarray(v,dtype=np.int64).tobytes()).hexdigest()
with rt.threadpool_limits(limits=1):
 for item in items:
  inv=next(x for x in base['inventory'] if x['dataset']==item['dataset'] and x['task']==item['task']);path=D/inv['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==inv['sha256']
  X,y=rt.load(path,item['task']);outer,test,_,_=rt.split_data(X,y,item['task'])
  for seed in SEEDS:
   it,iv=rt.train_test_split(outer,test_size=.25,random_state=seed,stratify=None if item['task']=='regression' else y.iloc[outer])
   if len(it)>4000:it,_=rt.train_test_split(it,train_size=4000,random_state=seed+100,stratify=None if item['task']=='regression' else y.iloc[it])
   assert not(set(it)|set(iv))&set(test) and not set(it)&set(iv)
   A,B,yt,yv,ym,ys,labels=rt.prep(X,y,item['task'],it,iv,item['mode'])
   splits.append({'cell':item['cell'],'seed':seed,'train_index_sha256':h(it),'validation_index_sha256':h(iv),'reserved_outer_test_index_sha256':h(test),'n_train':len(it),'n_validation':len(iv),'source_sha256':inv['sha256']})
   order=list(METHODS);np.random.default_rng(seed).shuffle(order)
   for method in order:
    ci=item[method+'_frozen_config'];r={'cell':item['cell'],'dataset':item['dataset'],'task':item['task'],'mode':item['mode'],'seed':seed,'method':method,'config':ci,'n_train':len(it),'n_validation':len(iv),'n_features':A.shape[1],'status':'success'}
    try:r.update(rt.fit_one(method,item['task'],base['configs'][ci],seed,A,B,yt,yv,ym,ys))
    except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
    rows.append(r);emit('fits.jsonl',r)
  print('EXPANDED',item['cell'],'done',flush=True)
summary=[]
for item in items:
 rs=[r for r in rows if r['cell']==item['cell'] and r['status']=='success'];r={k:item[k] for k in ['cell','dataset','task','mode','historically_better_than_xgb']};r['repeats']=len(SEEDS)
 r['n_train']=rs[0]['n_train'];r['n_validation']=rs[0]['n_validation'];r['n_features']=rs[0]['n_features']
 if not all(sum(x['method']==m for x in rs)==len(SEEDS) for m in METHODS):r['grade']='incomplete';summary.append(r);continue
 for m in METHODS:
  z=[x for x in rs if x['method']==m];r[m+'_primary_mean']=statistics.mean(x['primary'] for x in z);r[m+'_primary_sd']=statistics.stdev(x['primary'] for x in z);r[m+'_fit_wall_median']=statistics.median(x['fit_wall_sec'] for x in z);r[m+'_fit_cpu_median']=statistics.median(x['fit_cpu_sec'] for x in z)
  if item['task']!='regression':r[m+'_logloss_mean']=statistics.mean(x['logloss'] for x in z)
 for m in ['XGB','LGB_constant','LGB_linear']:
  gains=[]
  for seed in SEEDS:
   c=next(x['primary'] for x in rs if x['seed']==seed and x['method']=='CrossGB');b=next(x['primary'] for x in rs if x['seed']==seed and x['method']==m)
   gains.append(c-b if item['task']=='binary' else (b-c)/b)
  r['mean_paired_gain_vs_'+m]=statistics.mean(gains);r['sd_paired_gain_vs_'+m]=statistics.stdev(gains);r['wins_vs_'+m]=sum(x>0 for x in gains);r['fit_ratio_vs_'+m]=r['CrossGB_fit_wall_median']/r[m+'_fit_wall_median']
 threshold=.002 if item['task']=='binary' else .01
 stable=r['mean_paired_gain_vs_XGB']>=threshold and r['wins_vs_XGB']>=4
 near=all(r['mean_paired_gain_vs_'+m]>0 and r['wins_vs_'+m]>=3 for m in ['LGB_constant','LGB_linear'])
 r['grade']='A' if stable and near else ('B' if stable else ('C' if r['mean_paired_gain_vs_XGB']>0 else 'D'))
 summary.append(r)
for name,data in [('summary.csv',summary),('splits.csv',splits)]:
 keys=list(dict.fromkeys(k for row in data for k in row))
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
(O/'verification.json').write_text(json.dumps({'datasets':len(NAMES),'cells':len(items),'fit_attempts':len(rows),'success':sum(r['status']=='success' for r in rows),'formal_outer_test_evaluations':0,'wall_sec':time.perf_counter()-start},indent=2)+'\n')
for r in summary:print(r['grade'],r['cell'],'gainX',r.get('mean_paired_gain_vs_XGB'),'gainLinear',r.get('mean_paired_gain_vs_LGB_linear'),'wins',r.get('wins_vs_XGB'),r.get('wins_vs_LGB_linear'),flush=True)
