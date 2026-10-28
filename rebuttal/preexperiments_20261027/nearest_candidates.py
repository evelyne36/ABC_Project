"""Bounded new-UCI screen with frozen choices, reserved test, and author neighbours."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,importlib.util
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
import numpy as np
if len(sys.argv)>1 and sys.argv[1]=='--child':
 method,inp,out,cfg,seed=sys.argv[2:];cf=json.loads(cfg);seed=int(seed);data=np.load(inp)
 from threadpoolctl import threadpool_limits
 if method=='PL':
  sys.path.insert(0,str(WORK/'local_sync/gbdtpl_runtime'))
  from baselines.estimators import GBDTPLNative
  m=GBDTPLNative(cf,seed)
 else:
  sys.path.insert(0,str(WORK/'local_sync/author_linx'));from linxgb import linxgb
  m=linxgb(**cf,random_state=seed,nthread=1)
 with threadpool_limits(limits=1):
  w=time.perf_counter();c=time.process_time();m.fit(data['A'],data['yt']);fw=time.perf_counter()-w;fc=time.process_time()-c
  w=time.perf_counter();pred=m.predict(data['B'])*float(data['ys'])+float(data['ym']);pw=time.perf_counter()-w
  rmse=float(np.sqrt(np.mean((pred-data['yv'])**2)))
 if not np.isfinite(rmse):raise ValueError('nonfinite prediction')
 Path(out).write_text(json.dumps(dict(status='success',primary=rmse,fit_wall_sec=fw,fit_cpu_sec=fc,predict_wall_sec=pw)))
 raise SystemExit
import model_runtime as rt
rt.initialize(WORK/'check_codereview0701/beforereview/xgboost-master/python-package',Path('/opt/anaconda3/envs/llm/lib/python3.10/site-packages/lightgbm'),HERE/'mpl-cache')
O=HERE/'expansion_02';O.mkdir(exist_ok=False)
MAN=json.loads((WORK/'local_sync/new_tabular_data/manifest.json').read_text())
GRID={}
for method in ['XGB','CrossGB','LGB_constant','LGB_linear']:
 GRID[method]=[dict(depth=3,ridge=1.,p=1.,trees=100,lr=.1),dict(depth=3,ridge=10.,p=.5,trees=100,lr=.1),dict(depth=2,ridge=1.,p=1.,trees=300,lr=.05),dict(depth=4,ridge=1.,p=.25,trees=100,lr=.1)]
GRID['PL']=[dict(num_trees=100,max_depth=3,num_leaves=8,l2_reg=l,max_var=v,leaf_type=t,min_sum_hessians=5.,learning_rate=.1) for l,v,t in [(.1,3,'half_additive'),(1.,5,'half_additive'),(.1,5,'linear'),(10.,5,'linear')]]
GRID['LinXGBoost']=[dict(n_estimators=t,max_depth=3,lbda=l,min_samples_leaf=5,learning_rate=.3) for t,l in [(5,2.),(5,20.),(15,2.),(15,20.)]]
PLAN=dict(datasets=MAN,grid=GRID,discovery_seeds=[90101,90102],validation_seeds=[90201,90202,90203],train_cap=1200,split='seed42 first 5-fold test reserved; discovery 75% of remaining pool; a fixed 25% audit (seed78000) excluded from discovery. Two overlapping 75/25 discovery splits; three audit training subsamples/shuffles reuse one fixed audit.',selection='minimum mean discovery RMSE per method; failures remain; candidate must succeed on both discovery splits; no selection using audit labels',timeout_sec_per_author_fit=15,all_results_reported=True,formal_test_evaluations=0,provenance_note='Servo is simulation; three concrete targets are dependent tasks, not three independent datasets',timing_note='local ARM exploratory; PL uses scalar compatibility and API fit includes CSV I/O; Lin is author Python; no universal implementation-independent speed claims')
(O/'frozen_plan.json').write_text(json.dumps(PLAN,indent=2)+'\n')
ROWS=[];SPLITS=[]
def digest(a):return hashlib.sha256(np.asarray(a,dtype=np.int64).tobytes()).hexdigest()
def fit(method,cf,seed,A,B,yt,yv,ym,ys):
 if method in ['PL','LinXGBoost']:
  inp=O/'worker_input.npz';out=O/'worker_result.json';out.unlink(missing_ok=True);np.savez(inp,A=A,B=B,yt=yt,yv=yv,ym=ym,ys=ys)
  w=time.perf_counter();p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',method,str(inp),str(out),json.dumps(cf),str(seed)],capture_output=True,text=True,timeout=15)
  if p.returncode:raise RuntimeError(p.stderr[-800:])
  return dict(**json.loads(out.read_text()),process_wall_including_imports=time.perf_counter()-w)
 m=rt.make(method,'regression',cf,seed,cf['trees']);m.set_params(learning_rate=cf['lr']);w=time.perf_counter();c=time.process_time();m.fit(A,yt);fw=time.perf_counter()-w;fc=time.process_time()-c;w=time.perf_counter()
 with rt.warnings.catch_warnings():rt.warnings.simplefilter('ignore');pred=m.predict(B)*ys+ym
 val=float(np.sqrt(np.mean((pred-yv)**2)))
 if not np.isfinite(val):raise ValueError('nonfinite prediction')
 if method=='CrossGB':assert 'grow_metagb_histmaker' in m.get_booster().save_config() or '"enable_metagb":"1"' in m.get_booster().save_config()
 return dict(status='success',primary=val,fit_wall_sec=fw,fit_cpu_sec=fc,predict_wall_sec=time.perf_counter()-w)
def run(name,phase,seed,choices,X,y,it,iv,test):
 assert not (set(it)|set(iv))&set(test) and not set(it)&set(iv)
 A,B,yt,yv,ym,ys,_=rt.prep(X,y,'regression',it,iv,'onehot')
 SPLITS.append(dict(dataset=name,phase=phase,seed=seed,n_train=len(it),n_validation=len(iv),train_sha256=digest(it),validation_sha256=digest(iv),reserved_test_sha256=digest(test)))
 order=list(choices);np.random.default_rng(seed).shuffle(order)
 for method,ci in order:
  r=dict(dataset=name,phase=phase,seed=seed,method=method,config=ci,n_train=len(it),n_validation=len(iv),n_features=A.shape[1])
  try:r.update(fit(method,GRID[method][ci],seed,A,B,yt,yv,ym,ys))
  except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
  ROWS.append(r)
  with (O/'fits.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
CHOICES=[]
with rt.threadpool_limits(limits=1):
 for entry in MAN:
  name=entry['dataset'];path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'];X,y=rt.load(path,'regression');outer,test,disc,audit=rt.split_data(X,y,'regression')
  for seed in PLAN['discovery_seeds']:
   it,iv=rt.train_test_split(disc,test_size=.25,random_state=seed)
   if len(it)>1200:it=rt.np.random.default_rng(seed).choice(it,1200,replace=False)
   run(name,'discovery',seed,[(m,i) for m in GRID for i in range(len(GRID[m]))],X,y,it,iv,test)
  choices=[]
  for m in GRID:
   scores=[]
   for ci in range(len(GRID[m])):
    rs=[r for r in ROWS if r['dataset']==name and r['phase']=='discovery' and r['method']==m and r['config']==ci]
    if len(rs)==2 and all(r['status']=='success' for r in rs):scores.append((float(np.mean([r['primary'] for r in rs])),ci))
   if scores:choices.append((m,min(scores)[1]))
  CHOICES.append(dict(dataset=name,choices=choices));(O/'frozen_choices.json').write_text(json.dumps(CHOICES,indent=2)+'\n')
  for seed in PLAN['validation_seeds']:
   it=np.random.default_rng(seed).permutation(disc)[:1200];run(name,'audit',seed,choices,X,y,it,audit,test)
  print('FINISHED',name,'fits',len(ROWS),flush=True)
rt.pd.DataFrame(ROWS).to_csv(O/'fits.csv',index=False);rt.pd.DataFrame(SPLITS).to_csv(O/'splits.csv',index=False)
s=rt.pd.DataFrame(ROWS);valid=s[s.status=='success'];summary=valid.groupby(['dataset','phase','method']).agg(rmse=('primary','mean'),rmse_sd=('primary','std'),fit_wall=('fit_wall_sec','median'),fit_cpu=('fit_cpu_sec','median'),fits=('primary','size')).reset_index();summary.to_csv(O/'summary.csv',index=False)
(O/'verification.json').write_text(json.dumps(dict(status='complete',attempts=len(ROWS),success=int((s.status=='success').sum()),failed=int((s.status!='success').sum()),formal_test_evaluations=0,audit_caveat='three repeated fits on one fixed audit; not three independent splits'),indent=2)+'\n')
print(summary[summary.phase=='audit'].to_string(index=False),flush=True)
