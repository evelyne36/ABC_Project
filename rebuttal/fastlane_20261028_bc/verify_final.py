#!/usr/bin/env python3
"""Recompute new confirmation RMSE and paired accuracy/cost; never select models."""
import argparse,collections,csv,hashlib,json,pathlib,statistics
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
HOSTS=['cpu1_pool.zjuici.com','cpu2_pool.zjuici.com','cpu4_pool.zjuici.com','cpu5_pool.zjuici.com','cpu6_pool.zjuici.com','cpu_Calculator','TabularData_Processor']
def ah(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def output(p,rows):
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def checked_rmse(prediction,truth,reported):
 prediction=np.asarray(prediction);truth=np.asarray(truth)
 if prediction.shape!=truth.shape or not np.isfinite(prediction).all() or not np.isfinite(truth).all():raise ValueError('Nonfinite or wrong-shaped prediction/target')
 rmse=float(np.sqrt(np.mean((prediction-truth)**2)))
 if not np.isfinite(rmse) or not np.isfinite(reported) or reported<0:raise ValueError('Invalid metric')
 return rmse,abs(rmse-reported)
def positive_seconds(value):
 if not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:raise ValueError('Timing must be finite and positive')
 return value
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['B2','C'],required=True);p.add_argument('--results',nargs='+',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 recipes={'S1_sparse':json.loads((BASE/'B2_choice.json').read_text())} if a.stage=='B2' else json.loads((BASE/'inputs/c_frozen_choices.json').read_text())['cases']
 expected={(c,m,s) for c in recipes for m in list(recipes[c]['choices'])+['LeafOnly'] for s in range(99001,99006)}
 records={};rows=[];errors=[];maxdelta=0.
 for root in a.results:
  for f in root.rglob('result.json'):
   if 'final' not in f.parts:continue
   pos=f.parts.index('final');case=f.parts[pos-1];method=f.parts[pos+1];seed=int(f.parts[pos+2][1:]);key=(case,method,seed)
   if key not in expected:raise RuntimeError('Unexpected fit '+str(key))
   if key in records:raise RuntimeError('Duplicate fit '+str(key))
   records[key]=None;error_count=len(errors)
   r=json.loads(f.read_text());params=recipes[case]['choices']['CrossGB' if method=='LeafOnly' else method]['params']
   if r.get('method')!=method or r.get('seed')!=seed:errors.append(str(key)+': body identity mismatch')
   expected_worker=hashlib.sha256((BASE.parent/'fastlane_20261027/worker.py').read_bytes()).hexdigest()
   if r.get('worker_sha256')!=expected_worker or r.get('threads_per_fit')!=1:errors.append(str(key)+': worker/thread mismatch')
   if r.get('requested_params')!=params:errors.append(str(key)+': params mismatch')
   if r.get('status')!='ok':errors.append(str(key)+': '+r.get('status','missing'));records[key]=None;continue
   if not r.get('test_labels_used'):errors.append(str(key)+': missing explicit test marker')
   with np.load(BASE/'inputs/data'/f'{case}__s{seed}.npz') as d,np.load(f.parent/'predictions.npz') as pred:
    values={}
    for split in ['train','val','test']:
     truth=d['y_'+split];value=np.asarray(pred[split]);idx=d[split]
     if not np.array_equal(pred[split+'_indices'],idx):errors.append(str(key)+': index mismatch '+split)
     if r['split_hashes'][split]!=ah(idx) or r['target_hashes'][split]!=ah(truth):errors.append(str(key)+': split/target hash mismatch '+split)
     try:rmse,delta=checked_rmse(value,truth,r['metrics'][split]['rmse'])
     except (ValueError,TypeError):errors.append(str(key)+': invalid prediction/metric '+split);continue
     maxdelta=max(maxdelta,delta)
     if delta>1e-10:errors.append(str(key)+': RMSE mismatch '+split)
     values[split]=rmse
    host=next((x for x in HOSTS if x in f.parts),'unknown')
    if host=='unknown':errors.append(str(key)+': missing host identity')
    try:
     positive_seconds(r['timings']['fit_wall_sec']);positive_seconds(r['timings']['predict_test_wall_sec'])
    except (ValueError,KeyError):errors.append(str(key)+': invalid timing')
    if len(errors)>error_count:continue
    item={'case':case,'method':method,'seed':seed,'host':host,'train_rmse':values['train'],'val_rmse':values['val'],'test_rmse':values['test'],'fit_wall_sec':r['timings']['fit_wall_sec'],'predict_test_wall_sec':r['timings']['predict_test_wall_sec'],'params_sha256':hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest(),'source':str(f)}
    rows.append(item);records[key]=item
 for key in expected-set(records):errors.append(str(key)+': absent')
 pairs=[]
 for case in recipes:
  for baseline in list(recipes[case]['choices']):
   if baseline=='CrossGB':continue
   for seed in range(99001,99006):
    c=records.get((case,'CrossGB',seed));b=records.get((case,baseline,seed))
    if not c or not b:continue
    if c['host']!=b['host']:errors.append(f'{case}/{seed}: paired methods on different hosts')
    win=c['test_rmse']<b['test_rmse'];fit=b['fit_wall_sec']/c['fit_wall_sec'];pred=b['predict_test_wall_sec']/c['predict_test_wall_sec']
    pairs.append({'case':case,'baseline':baseline,'seed':seed,'rmse_improvement':(b['test_rmse']-c['test_rmse'])/b['test_rmse'],'accuracy_win':win,'fit_speedup':fit,'predict_speedup':pred,'accuracy_and_fit_win':win and fit>1,'accuracy_fit_predict_win':win and fit>1 and pred>1,'host':c['host']})
 summaries=[]
 for case in recipes:
  for method in list(recipes[case]['choices'])+['LeafOnly']:
   rs=[r for r in rows if r['case']==case and r['method']==method];v=[r['test_rmse'] for r in rs]
   summaries.append({'case':case,'method':method,'n':len(v),'test_rmse_mean':statistics.mean(v) if v else None,'test_rmse_sd':statistics.stdev(v) if len(v)>1 else None,'train_rmse_mean':statistics.mean(r['train_rmse'] for r in rs) if rs else None,'val_rmse_mean':statistics.mean(r['val_rmse'] for r in rs) if rs else None})
 ratios=[]
 for case in recipes:
  for method in recipes[case]['choices']:
   ps=[r for r in pairs if r['case']==case and r['baseline']==method]
   if ps:ratios.append({'case':case,'baseline':method,'paired_n':len(ps),'accuracy_wins':sum(r['accuracy_win'] for r in ps),'accuracy_and_fit_wins':sum(r['accuracy_and_fit_win'] for r in ps),'accuracy_fit_predict_wins':sum(r['accuracy_fit_predict_win'] for r in ps),'fit_speedup_paired_median':statistics.median(r['fit_speedup'] for r in ps),'predict_speedup_paired_median':statistics.median(r['predict_speedup'] for r in ps)})
 output(a.output/'folds.csv',rows);output(a.output/'summary.csv',summaries);output(a.output/'paired.csv',pairs);output(a.output/'paired_summary.csv',ratios)
 audit={'stage':a.stage,'expected_fits':len(expected),'observed_fits':len(records),'successful_fits':len(rows),'rmse_values_recomputed':3*len(rows),'max_recomputation_error':maxdelta,'errors':errors,'scope':'five independent new synthetic seeds; raw timings retained per host, summarize paired ratios only; frozen recipes never changed by this script'}
 (a.output/'verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
