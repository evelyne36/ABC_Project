"""Finite-sample local-slope hypotheses, matched negative controls, full neighbour screen."""
from pathlib import Path
import os,sys,json,time,subprocess,importlib.util
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
import model_runtime as rt
rt.initialize(WORK/'check_codereview0701/beforereview/xgboost-master/python-package',Path('/opt/anaconda3/envs/llm/lib/python3.10/site-packages/lightgbm'),HERE/'mpl-cache');np=rt.np
spec=importlib.util.spec_from_file_location('mechanisms',HERE.parent/'analysis_20261027/synthetic_mechanism_suite.py');gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
O=HERE/'synthetic_nearest_02';O.mkdir(exist_ok=False)
CASES=[dict(family=f,n=240,d=32,regions=2) for f in ['sparse_switch','dense_switch','offset_switch','step_control']]+[dict(family='sparse_switch',n=600,d=64,regions=4)]
GRID={m:[dict(depth=3,ridge=1.,p=1.,trees=1,lr=1.),dict(depth=3,ridge=1.,p=1.,trees=64,lr=.1),dict(depth=3,ridge=10.,p=1.,trees=64,lr=.1),dict(depth=3,ridge=.1,p=1.,trees=16,lr=.3)] for m in ['XGB','CrossGB','LGB_linear']}
GRID['PL']=[dict(num_trees=t,max_depth=3,num_leaves=8,l2_reg=l,max_var=5,leaf_type=k,min_sum_hessians=5.,learning_rate=lr) for t,l,k,lr in [(1,1.,'half_additive',1.),(64,1.,'half_additive',.1),(64,.1,'linear',.1),(64,10.,'linear',.1)]]
GRID['LinXGBoost']=[dict(n_estimators=t,max_depth=3,lbda=l,min_samples_leaf=5,learning_rate=lr) for t,l,lr in [(1,2.,1.),(4,2.,.3),(4,20.,.3),(4,200.,.3)]]
for m in ['XGB','LGB_linear']:
 GRID[m]+=[dict(depth=3,ridge=1.,p=1.,trees=1024,lr=.1),dict(depth=3,ridge=10.,p=1.,trees=1024,lr=.1)]
GRID['PL']+=[dict(num_trees=300,max_depth=3,num_leaves=8,l2_reg=l,max_var=5,leaf_type='half_additive',min_sum_hessians=5.,learning_rate=.1) for l in [.1,10.]]
PLAN=dict(cases=CASES,grids=GRID,discovery_seeds=[91001,91002],fresh_data_validation_seeds=[92001,92002,92003],noise_sigma=.3,split='75/25 within each independently generated dataset; same X, noise, train/validation indices across matched families',selection='per case and method minimum average development RMSE; require success on both development seeds; all failures and controls retained',timeout_per_author_fit_sec=12,primary='RMSE, lower better; no universal success threshold or expected winner guaranteed',hypothesis='many irrelevant predictors and small local samples favour sparse scalar leaves; path-only linear features may miss off-path slopes; affine neighbours can recover true signal and may outperform',limitation='local ARM PL scalar compatibility and Lin author Python; compare local error/time points, not implementation-independent speed')
(O/'frozen_plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');ROWS=[];CHOICES=[]
PLAN['predeclared_sparse_target_rmse']=.75
PLAN['target_success_rule']='RMSE <= .75 on every one of the three fresh data seeds, sparse cases only; no threshold relaxation'
(O/'frozen_plan.json').write_text(json.dumps(PLAN,indent=2)+'\n')
def cid(case):return '__'.join([case['family'],f"n{case['n']}",f"d{case['d']}",f"r{case['regions']}"])
def fit(m,cf,seed,A,B,yt,yv):
 if m in ['PL','LinXGBoost']:
  inp=O/'worker_input.npz';out=O/'worker_result.json';out.unlink(missing_ok=True);np.savez(inp,A=A,B=B,yt=yt,yv=yv,ym=0.,ys=1.)
  w=time.perf_counter();p=subprocess.run([sys.executable,str(HERE/'nearest_candidates.py'),'--child',m,str(inp),str(out),json.dumps(cf),str(seed)],capture_output=True,text=True,timeout=12)
  if p.returncode:raise RuntimeError(p.stderr[-800:])
  return dict(**json.loads(out.read_text()),process_wall_including_imports=time.perf_counter()-w)
 model=rt.make(m,'regression',cf,seed,cf['trees']);model.set_params(learning_rate=cf['lr']);w=time.perf_counter();c=time.process_time();model.fit(A,yt);fw=time.perf_counter()-w;fc=time.process_time()-c
 with rt.warnings.catch_warnings():rt.warnings.simplefilter('ignore');pred=model.predict(B)
 rmse=float(np.sqrt(np.mean((pred-yv)**2)))
 if not np.isfinite(rmse):raise ValueError('nonfinite metric')
 return dict(status='success',primary=rmse,fit_wall_sec=fw,fit_cpu_sec=fc)
def run(case,phase,seed,choices):
 arrays,_=gen.generate(**case,seed=seed,noise=.3);it,iv=rt.train_test_split(np.arange(case['n']),test_size=.25,random_state=seed);A=arrays['X'][it];B=arrays['X'][iv];yt=arrays['y'][it];yv=arrays['y'][iv]
 order=list(choices);np.random.default_rng(seed).shuffle(order)
 for m,ci in order:
  r=dict(case=cid(case),**case,phase=phase,seed=seed,method=m,config=ci,n_train=len(it),n_validation=len(iv))
  try:r.update(fit(m,GRID[m][ci],seed,A,B,yt,yv))
  except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
  ROWS.append(r)
  with (O/'fits.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
with rt.threadpool_limits(limits=1):
 for case in CASES:
  for seed in PLAN['discovery_seeds']:run(case,'discovery',seed,[(m,i) for m in GRID for i in range(len(GRID[m]))])
  choices=[]
  for m in GRID:
   scores=[]
   for ci in range(len(GRID[m])):
    rs=[r for r in ROWS if r['case']==cid(case) and r['phase']=='discovery' and r['method']==m and r['config']==ci]
    if len(rs)==2 and all(r['status']=='success' for r in rs):scores.append((np.mean([r['primary'] for r in rs]),ci))
   if scores:choices.append((m,min(scores)[1]))
  CHOICES.append(dict(case=cid(case),choices=choices));(O/'frozen_choices.json').write_text(json.dumps(CHOICES,indent=2)+'\n')
  for seed in PLAN['fresh_data_validation_seeds']:run(case,'fresh_validation',seed,choices)
  print('FINISHED',cid(case),'fits',len(ROWS),flush=True)
s=rt.pd.DataFrame(ROWS);s.to_csv(O/'fits.csv',index=False);v=s[s.status=='success'];summary=v.groupby(['case','phase','method']).agg(rmse=('primary','mean'),rmse_sd=('primary','std'),fit_wall=('fit_wall_sec','median'),fit_cpu=('fit_cpu_sec','median'),fits=('primary','size')).reset_index();summary.to_csv(O/'summary.csv',index=False)
(O/'verification.json').write_text(json.dumps(dict(status='complete',attempts=len(ROWS),success=int((s.status=='success').sum()),failed=int((s.status!='success').sum()),fresh_independent_data_seeds=3),indent=2)+'\n')
print(summary[summary.phase=='fresh_validation'].to_string(index=False),flush=True)
