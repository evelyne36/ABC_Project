"""Baseline-favourable follow-up: shallow Lin regularization and deeper PL; same frozen seeds."""
from pathlib import Path
import os,sys,json,subprocess,time,importlib.util
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
HERE=Path(__file__).resolve().parent;O=HERE/'synthetic_nearest_02/baseline_stress';O.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('mechanisms',HERE.parent/'analysis_20261027/synthetic_mechanism_suite.py');gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
base=json.loads((O.parent/'frozen_plan.json').read_text());GRID={}
GRID['LinXGBoost']=[dict(n_estimators=t,max_depth=d,lbda=l,min_samples_leaf=10,learning_rate=lr) for d,l,t,lr in [(d,l,1,1.) for d in [1,2] for l in [.2,2.,20.]]+[(1,l,4,.3) for l in [.2,2.,20.]]]
GRID['PL']=base['grids']['PL']+[dict(num_trees=100,max_depth=5,num_leaves=32,l2_reg=l,max_var=5,leaf_type=t,min_sum_hessians=5.,learning_rate=.1) for l,t in [(.1,'half_additive'),(1.,'half_additive'),(1.,'linear')]]
plan=dict(case=dict(family='sparse_switch',n=240,d=32,regions=2),discovery_seeds=[91001,91002],fresh_validation_seeds=[92001,92002,92003],grid=GRID,selection='lowest two-development-seed mean RMSE per neighbour; choices frozen before follow-up evaluation. Original fresh seed results already observed, so this is adaptive baseline strengthening, not independent confirmation.',CrossGB='retain original frozen single-tree configuration and results; no extra tuning for ours',timeout_sec=12,limitations='follow-up makes baseline choices more favourable; no fresh seeds independent of all study design')
(O/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[];choices=[]
def run(seed,phase,cs):
 ar,_=gen.generate(**plan['case'],seed=seed,noise=.3);it,iv=train_test_split(np.arange(240),test_size=.25,random_state=seed);inp=O/'input.npz';np.savez(inp,A=ar['X'][it],B=ar['X'][iv],yt=ar['y'][it],yv=ar['y'][iv],ym=0.,ys=1.)
 for method,ci in cs:
  out=O/'output.json';out.unlink(missing_ok=True);r=dict(method=method,config=ci,seed=seed,phase=phase)
  try:
   w=time.perf_counter();p=subprocess.run([sys.executable,str(HERE/'nearest_candidates.py'),'--child',method,str(inp),str(out),json.dumps(GRID[method][ci]),str(seed)],capture_output=True,text=True,timeout=12)
   if p.returncode:raise RuntimeError(p.stderr[-500:])
   r.update(json.loads(out.read_text()),process_wall_including_imports=time.perf_counter()-w)
  except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
  rows.append(r)
  with (O/'fits.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
for seed in plan['discovery_seeds']:run(seed,'discovery',[(m,i) for m in GRID for i in range(len(GRID[m]))])
for m in GRID:
 scores=[]
 for ci in range(len(GRID[m])):
  rs=[r for r in rows if r['method']==m and r['config']==ci]
  if len(rs)==2 and all(r['status']=='success' for r in rs):scores.append((np.mean([r['primary'] for r in rs]),ci))
 if scores:choices.append((m,min(scores)[1]))
(O/'frozen_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
for seed in plan['fresh_validation_seeds']:run(seed,'followup_validation',choices)
s=pd.DataFrame(rows);s.to_csv(O/'fits.csv',index=False);v=s[(s.phase=='followup_validation')&(s.status=='success')];summary=v.groupby('method').agg(rmse=('primary','mean'),rmse_sd=('primary','std'),fit_wall=('fit_wall_sec','median'),fit_cpu=('fit_cpu_sec','median'),n=('primary','size'));summary.to_csv(O/'summary.csv');(O/'verification.json').write_text(json.dumps(dict(attempts=len(rows),success=int((s.status=='success').sum()),failed=int((s.status!='success').sum()),status='complete'),indent=2)+'\n');print(summary.to_string(),flush=True)
