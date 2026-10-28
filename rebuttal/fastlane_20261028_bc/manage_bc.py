#!/usr/bin/env python3
"""Prepare immutable B/C jobs and validation-only selection from archived results."""
import argparse,datetime,hashlib,json,pathlib,tarfile
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
WORKSPACE=BASE.parents[1]
OLD=BASE.parent/'fastlane_20261027'
INPUTS=BASE/'inputs'
HOSTS=['cpu1_pool.zjuici.com','cpu2_pool.zjuici.com','cpu4_pool.zjuici.com','cpu5_pool.zjuici.com','cpu6_pool.zjuici.com','cpu_Calculator','TabularData_Processor']
REMOTE='/home/admin/crossgb_fastlane_20261027'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def package(stage,jobs):
 target=BASE/'dispatch'/stage
 if target.exists():raise RuntimeError('Immutable dispatch already exists')
 target.mkdir(parents=True);shards=[[] for _ in HOSTS]
 hosts=HOSTS if stage!='B2' else [h for h in HOSTS if h!='cpu6_pool.zjuici.com']+['cpu6_pool.zjuici.com']
 if stage=='B1':
  for i,job in enumerate(jobs):shards[i%len(HOSTS)].append(job)
 else:
  # All methods of a given case/seed run on one host/core for paired costs.
  groups={}
  for j in jobs:groups.setdefault((j['case'],j['seed']),[]).append(j)
  for i,group in enumerate(groups.values()):shards[i%len(HOSTS)].extend(group)
 commands=[]
 for host,shard in zip(hosts,shards):
  if not shard:continue
  source={j['data']:WORKSPACE/j['local_data'] for j in shard}
  p={'stage':stage,'host':host,'worker_sha256':sha(OLD/'worker.py'),'wall_budget_seconds':2700 if stage=='B1' else 1200,
     'jobs':shard,'data_sha256':{f:sha(src) for f,src in source.items()},'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  pf=target/(host+'.json');write(pf,p);archive=target/(host+'.tar.gz')
  with tarfile.open(archive,'w:gz') as t:
   for f in [BASE/'batch_runner.py',OLD/'worker.py']:t.add(f,arcname='fastlane_bc_v1/'+f.name)
   t.add(pf,arcname='fastlane_bc_v1/'+stage+'_plan.json')
   for path,src in source.items():
    t.add(src,arcname=path);meta=src.with_suffix('.json')
    if meta.exists():t.add(meta,arcname=str(pathlib.PurePosixPath(path).with_suffix('.json')))
  commands.append({'host':host,'action':'upload','local':str(archive),'remote':REMOTE,'timeout':120})
 write(target/'upload_jobs.json',commands)
 write(target/'manifest.json',{'stage':stage,'jobs':len(jobs),'hosts_used':len(commands),'plans':{h+'.json':sha(target/(h+'.json')) for h,s in zip(hosts,shards) if s}})
 print(json.dumps({'stage':stage,'jobs':len(jobs),'hosts_used':len(commands),'upload_jobs':str(target/'upload_jobs.json')}))
def b1():
 jobs=[]
 for j in read(INPUTS/'b1_missing_jobs.json')['jobs']:
  jobs.append({'id':f"Lin_c{j['candidate']:02}_s{j['seed']}",'case':j['case'],'method':j['method'],'candidate':j['candidate'],'seed':j['seed'],
    'params':j['params'],'data':'fastlane_data/'+pathlib.Path(j['data_relative_to_root']).name,'local_data':j['data_relative_to_root'],
    'relative_output':f"S1_sparse/search/LinXGBoost/c{j['candidate']:02}/s{j['seed']}",'evaluate_test':False,'timeout_seconds':300})
 package('B1',jobs)
def ah(value):
 value=np.ascontiguousarray(value);return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
def verify_development(path,record,method,candidate,seed,params):
 if record.get('method')!=method or record.get('seed')!=seed:raise RuntimeError('Development body identity mismatch: '+str(path))
 if record.get('worker_sha256')!=sha(OLD/'worker.py') or record.get('threads_per_fit')!=1:raise RuntimeError('Development worker/thread mismatch')
 if record.get('requested_params')!=params:raise RuntimeError('Development requested parameters mismatch')
 if record.get('test_labels_used') or record.get('metrics',{}).get('test'):raise RuntimeError('Development test exposure')
 with np.load(WORKSPACE/'local_sync/fastlane_data'/f'S1_sparse__s{seed}.npz') as data,np.load(path.parent/'predictions.npz') as pred:
  if 'test' in pred.files:raise RuntimeError('Development test predictions present')
  for split in ['train','val']:
   truth=data['y_'+split];indices=data[split];values=pred[split]
   if values.shape!=truth.shape or not np.isfinite(values).all() or not np.array_equal(pred[split+'_indices'],indices):raise RuntimeError('Invalid development predictions')
   if record['split_hashes'][split]!=ah(indices) or record['target_hashes'][split]!=ah(truth):raise RuntimeError('Development split/target mismatch')
   computed=float(np.sqrt(np.mean((values-truth)**2)));saved=record['metrics'][split]['rmse']
   if not np.isfinite(saved) or saved<0 or abs(computed-saved)>1e-10:raise RuntimeError('Development RMSE mismatch')
  score=record['val_rmse']
  if not np.isfinite(score) or score<0 or abs(score-record['metrics']['val']['rmse'])>1e-10:raise RuntimeError('Development selection metric mismatch')
def freeze(roots,allow_partial):
 pool=read(INPUTS/'b1_candidate_pool_20.json');ids=pool['candidate_order'];original=read(WORKSPACE/'local_sync/fastlane_results/cpu4_pool.zjuici.com/results/formal24_v1/S1_sparse/candidate_pool.json')
 records={};paths={}
 for r in read(INPUTS/'b1_reused_results.json')['records']:
  p=WORKSPACE/r['result_path']
  if sha(p)!=r['result_sha256']:raise RuntimeError('Archived result changed')
  records[(r['method'],r['candidate'],r['seed'])]=read(p);paths[(r['method'],r['candidate'],r['seed'])]=p
 for root in roots:
  for p in root.rglob('result.json'):
   parts=p.parts
   if 'search' not in parts:continue
   i=parts.index('search');method=parts[i+1];candidate=int(parts[i+2][1:]);seed=int(parts[i+3][1:]);key=(method,candidate,seed)
   if method!='LinXGBoost' or candidate not in ids or seed not in [97001,97002]:raise RuntimeError('Unexpected development record')
   if key in records:raise RuntimeError('Duplicate logical development fit')
   records[key]=read(p);paths[key]=p
 choices={};missing=[]
 for method in pool['methods']:
  scored=[]
  for ci in ids:
   vals=[]
   for seed in [97001,97002]:
    r=records.get((method,ci,seed),{})
    if r.get('test_labels_used') or r.get('metrics',{}).get('test'):raise RuntimeError('Development test exposure')
    if r.get('status')=='ok':
     verify_development(paths[(method,ci,seed)],r,method,ci,seed,original[method][ci])
     vals.append(r['val_rmse'])
   if len(vals)==2:scored.append((sum(vals)/2,ci))
   else:missing.append([method,ci])
  if not scored:raise RuntimeError('No complete candidate: '+method)
  score,ci=min(scored);choices[method]={'candidate':ci,'params':original[method][ci],'mean_development_val_rmse':score,'complete_candidates':len(scored),'unique_parameter_configs':len({json.dumps(original[method][i],sort_keys=True) for i in ids})}
 if missing and not allow_partial:raise RuntimeError('Incomplete candidates; no freeze yet. Use --allow-partial only after recorded budget completion: '+str(missing))
 out=BASE/'B2_choice.json'
 if out.exists():raise RuntimeError('Choice already frozen')
 write(out,{'case':'S1_sparse','choices':choices,'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':'Mean validation RMSE across two development seeds only','missing_candidates':missing,'new_seeds':[99001,99002,99003,99004,99005],'pool_sha256':sha(INPUTS/'b1_candidate_pool_20.json')})
 print(json.dumps(choices,indent=2))
def final(stage):
 if stage=='B2':cases={'S1_sparse':read(BASE/'B2_choice.json')}
 else:
  if not (BASE/'B_complete_verified.json').exists():raise RuntimeError('C must follow verified completion of B')
  cases=read(INPUTS/'c_frozen_choices.json')['cases']
 jobs=[]
 for case,record in cases.items():
  for seed in [99001,99002,99003,99004,99005]:
   methods={m:c['params'] for m,c in record['choices'].items()}
   methods['LeafOnly']=methods['CrossGB']
   for m,params in methods.items():jobs.append({'id':f'{case}_{m}_s{seed}','case':case,'method':m,'seed':seed,'params':params,
      'data':f'bc_inputs/data/{case}__s{seed}.npz','local_data':str((INPUTS/'data'/f'{case}__s{seed}.npz').relative_to(WORKSPACE)),
      'relative_output':f'{case}/final/{m}/s{seed}','evaluate_test':True,'save_model':m=='CrossGB','timeout_seconds':300})
 package(stage,jobs)
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['B1','freeze-B2','B2','C']);p.add_argument('--results',nargs='*',type=pathlib.Path,default=[]);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
 if a.stage=='B1':b1()
 elif a.stage=='freeze-B2':freeze(a.results,a.allow_partial)
 else:final(a.stage)
if __name__=='__main__':main()
