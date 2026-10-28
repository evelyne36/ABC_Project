"""Small matched synthetic error/time curves; reports all declared controls."""
from pathlib import Path
import sys,os,argparse,json,time,subprocess,importlib.util
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
if len(sys.argv)>1 and sys.argv[1]=='--lin-child':
 import numpy as np
 src,inp,out,trees,lr=sys.argv[2:];sys.path.insert(0,src)
 from linxgb import linxgb
 from threadpoolctl import threadpool_limits
 data=np.load(inp,allow_pickle=False)
 with threadpool_limits(limits=1):
  m=linxgb(n_estimators=int(trees),max_depth=3,lbda=2.,min_samples_leaf=10,learning_rate=float(lr),random_state=52001,nthread=1)
  w=time.perf_counter();c=time.process_time();m.fit(data['A'],data['yt']);fw=time.perf_counter()-w;fc=time.process_time()-c
  pred=m.predict(data['B']);rmse=float(np.sqrt(np.mean((pred-data['yv'])**2)))
 Path(out).write_text(json.dumps({'primary':rmse,'fit_wall_sec':fw,'fit_cpu_sec':fc,'status':'success'}))
 raise SystemExit(0)
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--xgboost-python-package',type=Path,required=True);p.add_argument('--lightgbm-package',type=Path,required=True);p.add_argument('--linx-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
import model_runtime as rt
rt.initialize(a.xgboost_python_package,a.lightgbm_package,O/'mpl-cache');np=rt.np
src=Path(__file__).resolve().parents[1]/'analysis_20261027/synthetic_mechanism_suite.py';spec=importlib.util.spec_from_file_location('mechs',src);gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
cases=[dict(family='sparse_switch',n=600,d=8,regions=2),dict(family='sparse_switch',n=600,d=8,regions=4),dict(family='sparse_switch',n=600,d=32,regions=4),dict(family='sparse_switch',n=2400,d=32,regions=4)]+[dict(family=f,n=600,d=32,regions=4) for f in ['dense_switch','offset_switch','step_control']]
plan={'cases':cases,'data_seeds':[52001,52002,52003],'learning_rates':[1.,.1],'tree_counts':[1,4,16,64],'native_methods':['XGB','CrossGB','LGB_constant','LGB_linear'],'shared_depth':3,'shared_ridge':1.,'predictor_fraction':1.,'noise_sigma':.3,'target_rmse_declared_before_fit':.45,'split':'75/25 within each generated dataset','all_methods_receive':'same full X; no gate/active-support oracle roles','Lin_scope':'pinned author implementation, only two n600 d8 sparse cases, 1/4 trees and learning_rate1; 20s timeout per fit; intercept unpenalized and lbda2 matches squared-loss Hessian scaling relative to native reg_lambda1','no_expected_winner_certified':True,'limitations':['local legacy builds','fixed small hyperparameter grid','no GBDT-PL run','finite simulation cases, not general superiority']}
(O/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
def emit(r):
 with (O/'fits.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
with rt.threadpool_limits(limits=1):
 for case in cases:
  case_id=f"{case['family']}__n{case['n']}__d{case['d']}__r{case['regions']}"
  for seed in plan['data_seeds']:
   arrays,truth=gen.generate(**case,seed=seed,noise=.3);it,iv=rt.train_test_split(np.arange(case['n']),test_size=.25,random_state=seed)
   A=arrays['X'][it];B=arrays['X'][iv];yt=arrays['y'][it];yv=arrays['y'][iv]
   for lr in plan['learning_rates']:
    for trees in plan['tree_counts']:
     order=list(plan['native_methods']);np.random.default_rng(seed+trees).shuffle(order)
     for method in order:
      # Reuse the validated factory; only learning_rate is changed equally for all four.
      cf=dict(depth=3,ridge=1.,p=1.);m=rt.make(method,'regression',cf,seed,trees)
      m.set_params(learning_rate=lr)
      row=dict(case=case_id,**case,seed=seed,method=method,trees=trees,learning_rate=lr,n_train=len(it),n_validation=len(iv),status='success')
      try:
       w=time.perf_counter();c=time.process_time();m.fit(A,yt);row.update(fit_wall_sec=time.perf_counter()-w,fit_cpu_sec=time.process_time()-c)
       with rt.warnings.catch_warnings():rt.warnings.simplefilter('ignore');pred=m.predict(B)
       row['primary']=float(np.sqrt(np.mean((pred-yv)**2)))
       if not np.isfinite(row['primary']):raise ValueError('nonfinite metric')
      except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e))
      emit(row)
   if case['family']=='sparse_switch' and case['d']==8:
    inp=O/f'{case_id}__s{seed}.npz';np.savez_compressed(inp,A=A,B=B,yt=yt,yv=yv)
    for trees in [1,4]:
     out=O/f'lin_{case_id}_s{seed}_t{trees}.json';row=dict(case=case_id,**case,seed=seed,method='LinXGBoost',trees=trees,learning_rate=1.,n_train=len(it),n_validation=len(iv))
     try:
      start=time.perf_counter();r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--lin-child',str(a.linx_source.resolve()),str(inp),str(out),str(trees),'1.0'],capture_output=True,text=True,timeout=20)
      if r.returncode:raise RuntimeError(r.stderr[-500:])
      row.update(json.loads(out.read_text()),process_wall_including_imports=time.perf_counter()-start)
     except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e))
     emit(row)
  print('SYNTHETIC',case_id,'done',flush=True)
(O/'status.json').write_text(json.dumps({'status':'complete','model_fit_attempts_expected':684},indent=2)+'\n')
