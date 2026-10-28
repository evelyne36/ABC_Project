"""Verify frozen native incremental fits and recompute all split metrics."""
from pathlib import Path
import json,csv,tarfile,hashlib,argparse,math
import numpy as np
I=Path(__file__).resolve().parent;ROOT=I.parents[1];RAW=ROOT/'local_sync/last_increment_results';OLD=I.parent/'advantage_exploration_20261028'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_entries():
 entries={}
 for stage in ['B','D']:
  for e in json.loads((I/(stage+'_inputs.json')).read_text()):entries[stage,e['case'],e['seed']]=(I/e['data'],I/e['metadata'])
 for kind in ['synthetic','real']:
  for e in json.loads((OLD/kind/'confirmation/manifest.json').read_text())['entries']:entries['C',e['case'],e['seed']]=(OLD/kind/'confirmation'/(e.get('data_npz') or e['npz']),OLD/kind/'confirmation'/(e.get('sidecar') or e['metadata']))
 return entries

def verify(stages):
 entries=load_entries();rows=[];status={}
 for stage in stages:
  d=I/'dispatch'/stage;manifest=json.loads((d/'manifest.json').read_text())
  for lane in manifest['lanes']:
   host=lane['host'];p=RAW/(host+'_'+stage+'.tar.gz');dest=RAW/host
   if not p.exists():continue
   dest.mkdir(exist_ok=True)
   with tarfile.open(p) as t:t.extractall(dest,filter='data')
   plan=json.loads((d/lane['plan']).read_text());base=dest/'results/last_increment_261028_v1'/stage
   if (base/'status.json').exists():status[stage+':'+host]=json.loads((base/'status.json').read_text())
   for j in plan['jobs']:
    f=base/j['relative_output']/'result.json'
    if not f.exists():continue
    r=json.loads(f.read_text());row=dict(stage=stage,case=j['case'],method=j['method'],seed=j['seed'],repeat=j.get('repeat',0),p=j.get('p'),status=r['status'],result=str(f.relative_to(ROOT)))
    if r['status']=='ok':
     data,meta=entries[stage,j['case'],j['seed']];assert sha(data)==plan['data_sha256'][j['data']];scale=json.loads(meta.read_text()).get('preprocessing',{}).get('y_scale',1.)
     assert r['requested_params']==j['params'] and r['method']==j['method'] and r['seed']==j['seed'] and r['worker_sha256']==plan['worker_sha256'] and r['threads_per_fit']==1;assert r['test_labels_used'] and r['evaluate_test']
     with np.load(data,allow_pickle=False) as a,np.load(f.parent/'predictions.npz',allow_pickle=False) as pred:
      for s in ['train','val','test']:
       assert np.array_equal(a[s],pred[s+'_indices']);score=float(np.sqrt(np.mean((pred[s]-a['y_'+s])**2)));assert math.isfinite(score) and abs(score-r[s+'_rmse'])<1e-8*max(1,score);row[s+'_rmse']=score*scale
     row.update(fit_wall=r['timings']['fit_wall_sec'],fit_cpu=r['timings']['fit_cpu_sec'],predict_test_wall=r['timings']['predict_test_wall_sec'],predict_test_cpu=r['timings']['predict_test_cpu_sec'])
    else:row.update(error=r.get('error'))
    rows.append(row)
 out=I/'reports';out.mkdir(exist_ok=True)
 if rows:
  keys=sorted(set().union(*(r.keys() for r in rows)))
  with (out/('native_'+'_'.join(stages)+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 (out/('native_'+'_'.join(stages)+'_verification.json')).write_text(json.dumps(dict(fits=len(rows),statuses={k:sum(r['status']==k for r in rows) for k in {r['status'] for r in rows}},workers=status),indent=2)+'\n')
 print('Verified native fits',len(rows),'statuses',{k:sum(r['status']==k for r in rows) for k in {r['status'] for r in rows}})
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stages',nargs='+');a=p.parse_args();verify(a.stages)
