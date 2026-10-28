"""One bounded follow-up: equal LR alternatives for two real cases and larger cheap baselines on synthetic."""
from pathlib import Path
import argparse,json,time,csv,importlib.util
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-project',type=Path,required=True);p.add_argument('--xgboost-python-package',type=Path,required=True);p.add_argument('--lightgbm-package',type=Path,required=True);p.add_argument('--scout',type=Path,required=True);p.add_argument('--synthetic',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();D=a.data_project.resolve();S=a.scout.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
import model_runtime as rt
rt.initialize(a.xgboost_python_package,a.lightgbm_package,O/'mpl-cache');np=rt.np
sp=json.loads((S/'frozen_plan.json').read_text());ds=list(csv.DictReader((S/'discovery_summary.csv').open()));synp=json.loads((a.synthetic/'frozen_plan.json').read_text())
plan={'reason':'Initial real LR0.1 curves can conflate per-tree cost with convergence rate; vary LR equally for all models. Initial synthetic baseline64-tree cap can underfit; expand cheap baselines instead of declaring unreachable error.','real_cells':['ilpd.csv::drop','vowel.csv::numeric'],'real_learning_rates':[.05,.2,.5],'real_trees':[20,60,180],'real_repetitions':2,'synthetic_baseline_methods':['XGB','LGB_constant','LGB_linear'],'synthetic_extra_trees':[128,256,1024],'synthetic_learning_rates':[.1,1.],'synthetic_cases':synp['cases'],'audit_labels_not_used':True,'further_expansion_after_this_batch':False}
(O/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
def emit(name,r):
 with (O/name).open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
with rt.threadpool_limits(limits=1):
 for cell in plan['real_cells']:
  item=next(x for x in ds if x['cell']==cell);inv=next(x for x in sp['inventory'] if x['dataset']==item['dataset']);X,y=rt.load(D/inv['path'],item['task']);_,test,disc,audit=rt.split_data(X,y,item['task']);pool=disc
  if len(pool)>1800:pool,_=rt.train_test_split(pool,train_size=1800,random_state=78101,stratify=None if item['task']=='regression' else y.iloc[pool])
  it,iv=rt.train_test_split(pool,test_size=.25,random_state=78101,stratify=None if item['task']=='regression' else y.iloc[pool]);A,B,yt,yv,ym,ys,labels=rt.prep(X,y,item['task'],it,iv,item['mode'])
  for lr in plan['real_learning_rates']:
   for trees in plan['real_trees']:
    for rep in range(2):
     order=['XGB','CrossGB','LGB_constant','LGB_linear'];np.random.default_rng(rep+trees).shuffle(order)
     for method in order:
      ci=int(item[method+'_frozen_config']);m=rt.make(method,item['task'],sp['configs'][ci],78101,trees);m.set_params(learning_rate=lr)
      r={'cell':cell,'dataset':item['dataset'],'task':item['task'],'mode':item['mode'],'method':method,'config':ci,'learning_rate':lr,'trees':trees,'repetition':rep,'status':'success'}
      try:
       w=time.perf_counter();c=time.process_time();m.fit(A,yt);r.update(fit_wall_sec=time.perf_counter()-w,fit_cpu_sec=time.process_time()-c)
       with rt.warnings.catch_warnings():rt.warnings.simplefilter('ignore');prob=m.predict_proba(B)
       r['primary']=float(rt.roc_auc_score(yv,prob[:,1])) if item['task']=='binary' else float(rt.log_loss(yv,prob,labels=np.arange(prob.shape[1])))
      except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
      emit('real_lr_frontiers.jsonl',r)
  print('LR FRONTIER',cell,'done',flush=True)
 src=Path(__file__).resolve().parents[1]/'analysis_20261027/synthetic_mechanism_suite.py';spec=importlib.util.spec_from_file_location('mechs',src);gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
 for case in plan['synthetic_cases']:
  cid=f"{case['family']}__n{case['n']}__d{case['d']}__r{case['regions']}"
  for seed in synp['data_seeds']:
   ar,_=gen.generate(**case,seed=seed,noise=.3);it,iv=rt.train_test_split(np.arange(case['n']),test_size=.25,random_state=seed);A=ar['X'][it];B=ar['X'][iv];yt=ar['y'][it];yv=ar['y'][iv]
   for lr in plan['synthetic_learning_rates']:
    for trees in plan['synthetic_extra_trees']:
     for method in plan['synthetic_baseline_methods']:
      m=rt.make(method,'regression',dict(depth=3,ridge=1.,p=1.),seed,trees);m.set_params(learning_rate=lr);r=dict(case=cid,**case,seed=seed,method=method,learning_rate=lr,trees=trees,status='success')
      try:
       w=time.perf_counter();c=time.process_time();m.fit(A,yt);r.update(fit_wall_sec=time.perf_counter()-w,fit_cpu_sec=time.process_time()-c)
       with rt.warnings.catch_warnings():rt.warnings.simplefilter('ignore');pred=m.predict(B)
       r['primary']=float(np.sqrt(np.mean((pred-yv)**2)))
      except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
      emit('synthetic_baseline_extensions.jsonl',r)
  print('EXTENDED SYNTHETIC',cid,'done',flush=True)
(O/'status.json').write_text(json.dumps({'status':'complete','model_fit_attempts_expected':522,'formal_test_evaluations':0},indent=2)+'\n')
