"""Evaluate frozen synthetic configurations on three fresh generated datasets per case."""
from pathlib import Path
import argparse,json,time,subprocess,sys,importlib.util
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--xgboost-python-package',type=Path,required=True);p.add_argument('--lightgbm-package',type=Path,required=True);p.add_argument('--linx-source',type=Path,required=True);p.add_argument('--synthetic',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False);S=a.synthetic.resolve()
import model_runtime as rt
rt.initialize(a.xgboost_python_package,a.lightgbm_package,O/'mpl-cache');np=rt.np
frozen=json.loads((S/'confirmation_frozen_choices.json').read_text());plan=json.loads((S/'frozen_plan.json').read_text());(O/'frozen_choices.json').write_text(json.dumps(frozen,indent=2)+'\n')
src=Path(__file__).resolve().parents[1]/'analysis_20261027/synthetic_mechanism_suite.py';spec=importlib.util.spec_from_file_location('mechs',src);gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
def emit(r):
 with (O/'fits.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
with rt.threadpool_limits(limits=1):
 for case in plan['cases']:
  cid=f"{case['family']}__n{case['n']}__d{case['d']}__r{case['regions']}";choices=[z for z in frozen['choices'] if z['case']==cid]
  for seed in frozen['new_data_seeds']:
   ar,_=gen.generate(**case,seed=seed,noise=.3);it,iv=rt.train_test_split(np.arange(case['n']),test_size=.25,random_state=seed);A=ar['X'][it];B=ar['X'][iv];yt=ar['y'][it];yv=ar['y'][iv]
   for choice in choices:
    method=choice['method'];trees=choice['trees'];lr=choice['learning_rate'];r=dict(case=cid,**case,method=method,trees=trees,learning_rate=lr,seed=seed,status='success')
    try:
     if method=='LinXGBoost':
      inp=O/f'{cid}__s{seed}.npz';np.savez_compressed(inp,A=A,B=B,yt=yt,yv=yv);out=O/f'lin_{cid}_s{seed}.json'
      cp=subprocess.run([sys.executable,str(Path(__file__).with_name('probe_synthetic.py').resolve()),'--lin-child',str(a.linx_source.resolve()),str(inp),str(out),str(trees),str(lr)],capture_output=True,text=True,timeout=20)
      if cp.returncode:raise RuntimeError(cp.stderr[-500:])
      r.update(json.loads(out.read_text()))
     else:
      m=rt.make(method,'regression',dict(depth=3,ridge=1.,p=1.),seed,trees);m.set_params(learning_rate=lr)
      w=time.perf_counter();c=time.process_time();m.fit(A,yt);r.update(fit_wall_sec=time.perf_counter()-w,fit_cpu_sec=time.process_time()-c)
      with rt.warnings.catch_warnings():rt.warnings.simplefilter('ignore');pred=m.predict(B)
      r['primary']=float(np.sqrt(np.mean((pred-yv)**2)))
    except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
    emit(r)
  print('FROZEN NEW-SEED CHECK',cid,'done',flush=True)
(O/'status.json').write_text(json.dumps({'status':'complete','fits_expected':90,'formal_real_test_evaluations':0},indent=2)+'\n')
