"""Fixed-route post-hoc validation-selected ridge sensitivity; sealed search."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,json,pathlib,hashlib,time
import numpy as np
import capacity_core as core
HERE=pathlib.Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
def fit_all(X,y,routes,trees,base,ridge,eta):
 cursor=np.full(len(y),base);fitted=core.anchor_entries(trees,X.shape[1]);features=list(range(X.shape[1]))
 for ti,entries in enumerate(fitted):
  residual=y-cursor
  for e in entries:
   mask=routes[:,ti]==e['leaf'];e['n_train']=int(mask.sum())
   if e['kind']=='constant':e['policy']='frozen_stored_constant';continue
   if not mask.any():e['policy']='no_training_rows_keep_stored_slope';continue
   theta,_=core.solve_leaf(X[mask],residual[mask],features,ridge,False)
   e.update(features=features,theta=(eta*theta).tolist(),affine=False,kind='refit',k=len(features),policy='training_only_all_features_refit')
  cursor+=core.contribution(X,routes[:,ti],entries)
 return fitted

def load_search(root,job):
 p=root/job['data'];assert sha(p)==job['search_data_sha256'];assert sha(root/job['model'])==job['model_sha256']
 with np.load(p,allow_pickle=False) as a:
  assert 'y' not in a.files and 'y_test' not in a.files
  X=a['X'].astype(np.float32).astype(np.float64);rows={s:a[s] for s in ['train','val','test']};y={s:a['y_'+s] for s in ['train','val']}
 raw=json.loads((root/job['model']).read_text())['learner'];trees=raw['gradient_booster']['model']['trees'];base=core.scalar(raw['learner_model_param']['base_score'])
 routes={s:core.crossgb_leaf_routes(X[rows[s]],trees) for s in ['train','val']}
 return X,rows,y,trees,base,routes

def search(root,job,out):
 out.mkdir(parents=True,exist_ok=False);X,rows,y,trees,base,routes=load_search(root,job);records=[];best={}
 with np.load(root/job['reference'],allow_pickle=False) as ref:
  for k in [1,2,3,'all']:
   chosen=None
   for multiplier in [.1,1.,10.,100.]:
    ridge=job['ridge']*multiplier;start=time.perf_counter();cpu=time.process_time()
    fitted=fit_all(X[rows['train']],y['train'],routes['train'],trees,base,ridge,job['eta']) if k=='all' else core.fit_variant(X[rows['train']],y['train'],routes['train'],trees,base,ridge,job['eta'],k,False)
    wall=time.perf_counter()-start;cpu=time.process_time()-cpu
    pred={s:core.predict_frozen(X[rows[s]],routes[s],fitted,base) for s in ['train','val']}
    metrics={s:rmse(pred[s],y[s]) for s in pred};assert all(np.isfinite(v) for v in metrics.values())
    replay=0.
    if multiplier==1. and k!='all':
     replay=max(float(np.max(np.abs(pred[s]-ref[f'slope_k{k}__{s}']))) for s in ['train','val']);assert replay<1e-8,replay
    row=dict(k=k,multiplier=multiplier,ridge=ridge,train_rmse=metrics['train'],val_rmse=metrics['val'],fit_wall=wall,fit_cpu=cpu,lambda1_replay_max_abs=replay)
    records.append(row)
    if chosen is None or metrics['val']<chosen['val_rmse']:
     chosen=dict(row);best[str(k)]=dict(choice=chosen,model=fitted)
    write(out/'progress.json',dict(case=job['case'],seed=job['seed'],completed=len(records),total=16,test_labels_used=False))
  models={k:x['model'] for k,x in best.items()};write(out/'models.json',models)
  write(out/'search.json',dict(case=job['case'],seed=job['seed'],status='ok',scale=job['scale'],records=records,choices={k:x['choice'] for k,x in best.items()},model_cache_sha256=sha(out/'models.json'),test_labels_used=False,original_model_sha256=job['model_sha256'],original_data_sha256=job['original_data_sha256'],core_sha256=sha(HERE/'capacity_core.py')))

def evaluate(root,job,search_dir,out,freeze):
 frozen=json.loads(freeze.read_text());key=f"{job['case']}/s{job['seed']}";r=json.loads((search_dir/'search.json').read_text());assert frozen['choices'][key]['search_sha256']==sha(search_dir/'search.json');assert r['model_cache_sha256']==sha(search_dir/'models.json')
 models=json.loads((search_dir/'models.json').read_text());X,rows,y,trees,base,routes=load_search(root,job)
 with np.load(root/job['test_data'],allow_pickle=False) as a:
  assert sha(root/job['test_data'])==job['test_data_sha256'];assert np.array_equal(a['test'],rows['test']);y['test']=a['y_test']
 routes['test']=core.crossgb_leaf_routes(X[rows['test']],trees);predictions={};metrics={};out.mkdir(parents=True,exist_ok=False)
 for k,model in models.items():
  metrics[k]={}
  for s in ['train','val','test']:
   pred=core.predict_frozen(X[rows[s]],routes[s],model,base);predictions[f'k{k}__{s}']=pred;metrics[k][s]=rmse(pred,y[s]);assert np.isfinite(metrics[k][s])
 np.savez_compressed(out/'predictions.npz',**predictions,**{s+'_indices':idx for s,idx in rows.items()})
 write(out/'result.json',dict(status='ok',case=job['case'],seed=job['seed'],metrics=metrics,choices=r['choices'],scale=job['scale'],freeze_sha256=sha(freeze),test_labels_used=True,scope='posthoc fixed-routing validation-selected ridge sensitivity; inherited routes and shared historical test; not independent new confirmation'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--plan',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--phase',choices=['search','evaluate'],default='search');p.add_argument('--search',type=pathlib.Path);p.add_argument('--freeze',type=pathlib.Path);a=p.parse_args();plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=False)
 assert sha(HERE/'capacity_core.py')==plan['core_sha256'];assert sha(__file__ and pathlib.Path(__file__))==plan['worker_sha256']
 core_id=sorted(os.sched_getaffinity(0))[0];os.sched_setaffinity(0,{core_id});start=time.monotonic()
 for ix,job in enumerate(plan['jobs']):
  write(a.output/'status.json',dict(status='running',phase=a.phase,completed=ix,total=len(plan['jobs']),pid=os.getpid(),current=job['case']+'/s'+str(job['seed'])))
  assert time.monotonic()-start<plan['wall_budget_seconds'],'A stage budget exhausted'
  dest=a.output/job['case']/('s'+str(job['seed']))
  if a.phase=='search':search(a.root,job,dest)
  else:evaluate(a.root,job,a.search/job['case']/('s'+str(job['seed'])),dest,a.freeze)
 write(a.output/'status.json',dict(status='complete',phase=a.phase,completed=len(plan['jobs']),total=len(plan['jobs']),pid=os.getpid(),elapsed=time.monotonic()-start))
if __name__=='__main__':main()
