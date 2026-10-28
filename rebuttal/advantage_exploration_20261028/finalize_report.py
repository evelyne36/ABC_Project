"""Aggregate verified confirmation, timing, paired outcomes, and provenance."""
import pathlib,json,csv,statistics,hashlib,datetime
from collections import defaultdict
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
METHODS=['CrossGB','XGB','LGB_constant','LGB_linear','PL','LinXGBoost','LeafOnly']
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=list(csv.DictReader((HERE/'confirmation_report/fits.csv').open()));times=list(csv.DictReader((HERE/'timing_report/fits.csv').open()))
assert len(rows)==70 and len(times)==42 and all(r['status']=='ok' for r in rows+times)
index={(r['case'],r['method'],int(r['seed'])):r for r in rows};summary=[];paired=[];timing=[]
for t in times:
 r=index[t['case'],t['method'],int(t['seed'])]
 for k in ['primary_test','val_rmse','train_rmse']:
  assert abs(float(t[k])-float(r[k]))<=1e-6*max(1,abs(float(r[k]))),(t['case'],t['method'],k)
for case in sorted({r['case'] for r in rows}):
 for method in METHODS:
  rs=sorted([r for r in rows if r['case']==case and r['method']==method],key=lambda r:int(r['seed']));assert len(rs)==5
  v=[float(r['primary_test']) for r in rs];val=[float(r['selection_value']) for r in rs];train=[float(r['train_rmse']) for r in rs]
  summary.append(dict(case=case,method=method,n=5,train_rmse_mean=statistics.mean(train),val_rmse_mean=statistics.mean(val),test_rmse_mean=statistics.mean(v),test_rmse_std=statistics.stdev(v),test_rmse_min=min(v),test_rmse_max=max(v)))
  ts=[t for t in times if t['case']==case and t['method']==method];assert len(ts)==3 and len({t['host'] for t in ts})==1
  timing.append(dict(case=case,method=method,host=ts[0]['host'],seed=int(ts[0]['seed']),repeats=3,fit_wall_median=statistics.median(float(t['fit_wall']) for t in ts),fit_wall_min=min(float(t['fit_wall']) for t in ts),fit_wall_max=max(float(t['fit_wall']) for t in ts),fit_cpu_median=statistics.median(float(t['fit_cpu']) for t in ts),predict_test_wall_median=statistics.median(float(t['predict_test_wall']) for t in ts),predict_test_cpu_median=statistics.median(float(t['predict_test_cpu']) for t in ts)))
  if method=='CrossGB':continue
  ours=[float(index[case,'CrossGB',int(r['seed'])]['primary_test']) for r in rs];rel=[1-a/b for a,b in zip(ours,v)]
  paired.append(dict(case=case,baseline=method,paired_n=5,cgb_accuracy_wins=sum(a<b for a,b in zip(ours,v)),relative_mean_rmse_reduction=1-statistics.mean(ours)/statistics.mean(v),mean_paired_rmse_reduction=statistics.mean(rel),min_paired_rmse_reduction=min(rel),max_paired_rmse_reduction=max(rel)))
for p in paired:
 a=next(t for t in timing if t['case']==p['case'] and t['method']=='CrossGB');b=next(t for t in timing if t['case']==p['case'] and t['method']==p['baseline'])
 assert a['host']==b['host'];p['fit_speedup_baseline_over_cgb']=b['fit_wall_median']/a['fit_wall_median'];p['predict_speedup_baseline_over_cgb']=b['predict_test_wall_median']/a['predict_test_wall_median']
OUT=HERE/'final_report';OUT.mkdir(exist_ok=True)
csvwrite(OUT/'confirmation_summary.csv',summary);csvwrite(OUT/'paired_comparisons.csv',paired);csvwrite(OUT/'exclusive_timing.csv',timing)
write(OUT/'verification.json',dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),development_fits=960,confirmation_fits=70,timing_fits=42,total_fits=1072,status='all_ok',test_predictions_recomputed=112,timing_replays_match_confirmation=True,confirmation_freeze_sha256=hashlib.sha256((HERE/'confirmation_freeze.json').read_bytes()).hexdigest(),no_additional_hpo_after_test=True))
for case in sorted({r['case'] for r in rows}):
 print(case)
 for p in paired:
  if p['case']==case:print(p['baseline'],'wins',p['cgb_accuracy_wins'],'RMSE reduction%',round(100*p['relative_mean_rmse_reduction'],3),'fit speedup',round(p['fit_speedup_baseline_over_cgb'],3),'predict speedup',round(p['predict_speedup_baseline_over_cgb'],3))
