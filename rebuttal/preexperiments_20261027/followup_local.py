"""Supplementary predeclared audit cases and exploratory fixed-configuration frontiers."""
from pathlib import Path
import argparse,os,json,csv,time,statistics,hashlib,sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-project',type=Path,required=True);p.add_argument('--xgboost-python-package',type=Path,required=True);p.add_argument('--lightgbm-package',type=Path,required=True);p.add_argument('--scout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();D=a.data_project.resolve();S=a.scout.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
import model_runtime as rt
rt.initialize(a.xgboost_python_package,a.lightgbm_package,O/'mpl-cache')
np,pd=rt.np,rt.pd;METHODS=['XGB','CrossGB','LGB_constant','LGB_linear']
base=json.loads((S/'frozen_plan.json').read_text());inventory=base['inventory'];configs=base['configs'];extra=json.loads((S/'supplementary_shortlist_frozen.json').read_text())['selected']
summary=list(csv.DictReader((S/'discovery_summary.csv').open()))
profile_cells=['kin8nm.csv::numeric','vowel.csv::numeric','diamonds.csv::drop','vehicle.csv::numeric','phoneme.csv::numeric']
plan={'supplementary_audit':extra,'profile_cells':profile_cells,'profile_selection':'frozen from discovery and historical leads; profile scoring uses discovery rows only, never locked-audit or formal-test rows','profile_trees':[10,30,100,300,900],'profile_repetitions':3,'profile_train_cap':1800,'model_config':'each method frozen from scout two development splits; only tree count varies','no_formal_test_evaluations':True,'randomness_note':'legacy CrossGB updater resets its internal C RNG to 1; repeated identical-data fits quantify timing noise, not independent data-seed uncertainty'}
(O/'frozen_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
def emit(name,r):
 with (O/name).open('a') as f:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
with rt.threadpool_limits(limits=1):
 for item in extra:
  inv=next(x for x in inventory if x['dataset']==item['dataset'] and x['task']==item['task']);X,y=rt.load(D/inv['path'],item['task']);_,test,disc,audit=rt.split_data(X,y,item['task'])
  for seed in [79101,79102,79103]:
   it=disc
   if len(it)>4000:it,_=rt.train_test_split(it,train_size=4000,random_state=seed,stratify=None if item['task']=='regression' else y.iloc[it])
   A,B,yt,yv,ym,ys,labels=rt.prep(X,y,item['task'],it,audit,item['mode'])
   for method in METHODS:
    ci=int(item[method+'_frozen_config']);r={'stage':'supplementary_locked_audit','cell':item['cell'],'dataset':item['dataset'],'task':item['task'],'mode':item['mode'],'method':method,'config':ci,'seed':seed,'n_train':len(it),'n_validation':len(audit),'n_features':A.shape[1],'status':'success'}
    try:r.update(rt.fit_one(method,item['task'],configs[ci],seed,A,B,yt,yv,ym,ys))
    except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
    emit('supplementary_audit.jsonl',r)
  print('EXTRA AUDIT',item['cell'],'done',flush=True)
 for cell in profile_cells:
  item=next(r for r in summary if r['cell']==cell);inv=next(x for x in inventory if x['dataset']==item['dataset'] and x['task']==item['task']);X,y=rt.load(D/inv['path'],item['task']);_,test,disc,audit=rt.split_data(X,y,item['task'])
  pool=disc
  if len(pool)>1800:pool,_=rt.train_test_split(pool,train_size=1800,random_state=78101,stratify=None if item['task']=='regression' else y.iloc[pool])
  it,iv=rt.train_test_split(pool,test_size=.25,random_state=78101,stratify=None if item['task']=='regression' else y.iloc[pool])
  assert not(set(it)|set(iv))&(set(test)|set(audit))
  A,B,yt,yv,ym,ys,labels=rt.prep(X,y,item['task'],it,iv,item['mode'])
  for trees in plan['profile_trees']:
   for rep in range(plan['profile_repetitions']):
    order=list(METHODS);np.random.default_rng(80000+rep+trees).shuffle(order)
    for method in order:
     ci=int(item[method+'_frozen_config']);r={'stage':'discovery_frontier','cell':cell,'dataset':item['dataset'],'task':item['task'],'mode':item['mode'],'method':method,'config':ci,'trees':trees,'repetition':rep,'n_train':len(it),'n_validation':len(iv),'n_features':A.shape[1],'status':'success'}
     try:r.update(rt.fit_one(method,item['task'],configs[ci],78101,A,B,yt,yv,ym,ys,trees=trees))
     except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
     emit('real_frontiers.jsonl',r)
  print('FRONTIER',cell,'done',flush=True)
(O/'status.json').write_text(json.dumps({'status':'complete','formal_test_evaluations':0},indent=2)+'\n')
