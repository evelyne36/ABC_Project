"""Recompute metrics and identities without exposing sealed labels during search."""
import argparse,pathlib,json,csv,hashlib,collections,tarfile,math
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];RAW=ROOT/'local_sync/advantage_results'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
def inspect(stages,output,freeze=False):
 rows=[];checked=0
 entries={}
 for kind in ['synthetic','real']:
  manifest=HERE/kind/'manifest.json'
  for e in json.loads(manifest.read_text())['entries']:entries[(e['case'],e['seed'])]=(kind,HERE/kind,e)
  cm=HERE/kind/'confirmation'/'manifest.json'
  if cm.exists():
   for e in json.loads(cm.read_text())['entries']:entries[(e['case'],e['seed'])]=(kind,HERE/kind/'confirmation',e)
 for stage in stages:
  directory=HERE/'dispatch' if stage=='quick' else HERE/'dispatch'/stage
  manifest=json.loads((directory/'manifest.json').read_text())
  archives={lane['host'] for lane in manifest['lanes']}
  for host in archives:
   archive=RAW/(host+'_'+stage+'.tar.gz');dest=RAW/host
   if archive.exists():
    dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as tf:tf.extractall(dest,filter='data')
  for lane in manifest['lanes']:
   host=lane['host'];base=RAW/host/'results/advantage_20261028_v1'/stage
   if stage=='quick':base=base/f'w{lane["lane"]}'
   plan=json.loads((directory/lane['plan']).read_text())
   for job in plan['jobs']:
    f=base/job['relative_output']/'result.json'
    if not f.exists():continue
    r=json.loads(f.read_text());row={k:job[k] for k in ['case','method','candidate','seed']};row.update(stage=stage,host=host,status=r['status'],val_rmse=None,selection_value=None,fit_wall=None,result_path=str(f.relative_to(ROOT)))
    if r['status']=='ok':
     kind,input_base,e=entries[job['case'],job['seed']];npz=input_base/(e.get('data_npz') or e['npz']);side=input_base/(e.get('sidecar') or e['metadata']);meta=json.loads(side.read_text());scale=meta.get('preprocessing',{}).get('y_scale',1.)
     assert digest(npz)==plan['data_sha256'][job['data']]
     assert r['method']==job['method'] and r['seed']==job['seed'];assert r['requested_params']==job['params'];assert r['worker_sha256']==plan['worker_sha256'];assert r['threads_per_fit']==1
     assert r['evaluate_test']==job['evaluate_test'];assert r['test_labels_used']==job['evaluate_test'];assert ('test' in r['metrics'])==job['evaluate_test']
     with np.load(npz,allow_pickle=False) as data,np.load(f.parent/'predictions.npz',allow_pickle=False) as pred:
      for split in ['train','val']+(['test'] if job['evaluate_test'] else []):
       assert np.array_equal(data[split],pred[split+'_indices']);actual=float(np.sqrt(np.mean((pred[split]-data['y_'+split])**2)));assert math.isfinite(actual);assert abs(actual-r[split+'_rmse'])<=1e-8*max(1,actual)
       assert set(pred.files)==set([x for s in (['train','val','test'] if job['evaluate_test'] else ['train','val']) for x in [s,s+'_indices']])
     row.update(repeat=job.get('repeat'),fit_cpu=r['timings'].get('fit_cpu_sec'),predict_test_wall=r['timings'].get('predict_test_wall_sec'),predict_test_cpu=r['timings'].get('predict_test_cpu_sec'),kind=kind,val_rmse=r['val_rmse'],selection_value=r['val_rmse']*scale,fit_wall=r['timings']['fit_wall_sec'],train_rmse=r['train_rmse']*scale,primary_test=r.get('test_rmse',0)*scale if job['evaluate_test'] else None);checked+=1
    else:row.update(error=r.get('error'),failure_stage=r.get('failure_stage'))
    rows.append(row)
 output.mkdir(exist_ok=True)
 if rows:
  keys=sorted(set().union(*(r.keys() for r in rows)))
  with (output/'fits.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 write(output/'verification.json',dict(fits_seen=len(rows),statuses=dict(collections.Counter(r['status'] for r in rows)),successful_predictions_recomputed=checked,stages=stages))
 finalists=json.loads((HERE/'finalist_freeze.json').read_text())['finalists'];choices={}
 for kind,case in finalists:
  choices[case]={}
  pools=json.loads((HERE/'frozen_candidate_pools.json').read_text())[kind]
  for method,configs in pools.items():
   if method=='LeafOnly':continue
   candidates=[]
   for ci in range(20):
    rs=[r for r in rows if r['case']==case and r['method']==method and r['candidate']==ci and r['stage'] in ['quick','hpo'] and r['status']=='ok']
    if len(rs)==2 and len({r['seed'] for r in rs})==2:candidates.append((sum(r['selection_value'] for r in rs)/2,ci))
   if candidates:
    score,ci=min(candidates);params=configs[ci];h=hashlib.sha256(json.dumps(params,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    choices[case][method]=dict(candidate=ci,params=params,params_sha256=h,mean_development_primary=score,complete_candidates=len(candidates))
 write(output/'choices.json',choices)
 for case,cs in choices.items():
  print(case,[(m,round(v['mean_development_primary'],6),v['candidate'],v['complete_candidates']) for m,v in cs.items()])
 print('VERIFIED',checked,'STATUSES',dict(collections.Counter(r['status'] for r in rows)))
 if freeze:
  import datetime
  assert all(len(cs)==6 and all(v['complete_candidates']==20 for v in cs.values()) for cs in choices.values()),'Incomplete HPO must be repaired or explicitly reviewed before freeze'
  freeze_file=HERE/'confirmation_freeze.json';assert not freeze_file.exists()
  write(freeze_file,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=list(choices),methods=list(next(iter(choices.values()))),choices=choices,exploration_protocol_sha256=digest(HERE/'exploration_protocol.json'),candidate_pool_sha256=digest(HERE/'frozen_candidate_pools.json'),development_report_sha256=digest(output/'fits.csv'),confirmation_test_seen=False))
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stages',nargs='+',default=['quick','hpo']);p.add_argument('--output',default='hpo_report');p.add_argument('--freeze',action='store_true');a=p.parse_args();inspect(a.stages,HERE/a.output,a.freeze)
