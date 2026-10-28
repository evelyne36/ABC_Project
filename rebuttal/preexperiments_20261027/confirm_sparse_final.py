"""Final independently generated seeds after adaptive baseline strengthening."""
from pathlib import Path
import os,sys,json,time,subprocess,importlib.util
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1];O=HERE/'synthetic_nearest_02/final_fresh';O.mkdir(exist_ok=False)
import model_runtime as rt
rt.initialize(WORK/'check_codereview0701/beforereview/xgboost-master/python-package',Path('/opt/anaconda3/envs/llm/lib/python3.10/site-packages/lightgbm'),O/'mpl-cache');np=rt.np
spec=importlib.util.spec_from_file_location('mechs',HERE.parent/'analysis_20261027/synthetic_mechanism_suite.py');gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
base=json.loads((O.parent/'frozen_plan.json').read_text());stress=json.loads((O.parent/'baseline_stress/frozen_plan.json').read_text());grids=base['grids']
cf={'CrossGB':grids['CrossGB'][0],'XGB':grids['XGB'][0],'LGB_linear':grids['LGB_linear'][3],'PL':grids['PL'][0],'LinXGBoost':stress['grid']['LinXGBoost'][4]}
plan=dict(case=dict(family='sparse_switch',n=240,d=32,regions=2),seeds=[93001,93002,93003,93004,93005],configs=cf,selection='freeze after previous adaptive work; Lin uses shallower setting that improved previous validation; ours unchanged; these five seeds not previously used',target_rmse=.75,primary='paired RMSE and local fit wall/CPU; 5 independent generated datasets; no broad significance claim',split='180 train, 60 validation; random_state equals data seed')
(O/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
with rt.threadpool_limits(limits=1):
 for seed in plan['seeds']:
  ar,_=gen.generate(**plan['case'],seed=seed,noise=.3);it,iv=rt.train_test_split(np.arange(240),test_size=.25,random_state=seed);A=ar['X'][it];B=ar['X'][iv];yt=ar['y'][it];yv=ar['y'][iv];methods=list(cf);np.random.default_rng(seed).shuffle(methods)
  for method in methods:
   r=dict(seed=seed,method=method,status='success')
   try:
    if method in ['PL','LinXGBoost']:
     inp=O/'input.npz';out=O/'output.json';out.unlink(missing_ok=True);np.savez(inp,A=A,B=B,yt=yt,yv=yv,ym=0.,ys=1.);w=time.perf_counter();p=subprocess.run([sys.executable,str(HERE/'nearest_candidates.py'),'--child',method,str(inp),str(out),json.dumps(cf[method]),str(seed)],capture_output=True,text=True,timeout=12)
     if p.returncode:raise RuntimeError(p.stderr[-600:])
     r.update(json.loads(out.read_text()),process_wall_including_imports=time.perf_counter()-w)
    else:
     c=cf[method];m=rt.make(method,'regression',c,seed,c['trees']);m.set_params(learning_rate=c['lr']);w=time.perf_counter();cpu=time.process_time();m.fit(A,yt);r.update(fit_wall_sec=time.perf_counter()-w,fit_cpu_sec=time.process_time()-cpu)
     with rt.warnings.catch_warnings():rt.warnings.simplefilter('ignore');pred=m.predict(B)
     r['primary']=float(np.sqrt(np.mean((pred-yv)**2)))
    if not np.isfinite(r['primary']):raise ValueError('nonfinite prediction')
   except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
   rows.append(r)
   with (O/'fits.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
s=rt.pd.DataFrame(rows);s.to_csv(O/'fits.csv',index=False);v=s[s.status=='success'];summary=v.groupby('method').agg(rmse=('primary','mean'),rmse_sd=('primary','std'),fit_wall=('fit_wall_sec','median'),fit_cpu=('fit_cpu_sec','median'),fits=('primary','size'),target_hits=('primary',lambda x:sum(x<=.75)));summary.to_csv(O/'summary.csv')
(O/'verification.json').write_text(json.dumps(dict(status='complete',attempts=len(rows),success=int((s.status=='success').sum()),failed=int((s.status!='success').sum()),independent_new_data_seeds=5,formal_real_test_evaluations=0),indent=2)+'\n');print(summary.to_string(),flush=True)
