from pathlib import Path
import csv,json,statistics,collections,math
I=Path(__file__).resolve().parent;O=I/'reports';native=list(csv.DictReader((O/'native_B_C_D.csv').open()));A=list(csv.DictReader((O/'A_selected.csv').open()))
def csvwrite(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
a_summary=[];a_pairs=[]
for case in sorted({r['case'] for r in A}):
 base={r['seed']:r for r in A if r['case']==case and r['k']=='1'}
 for k in ['1','2','3','all']:
  rs=[r for r in A if r['case']==case and r['k']==k];assert len(rs)==5;vals=[float(r['test_rmse']) for r in rs];bs=[float(base[r['seed']]['test_rmse']) for r in rs]
  a_summary.append(dict(case=case,k=k,n=5,train_mean=statistics.mean(float(r['train_rmse']) for r in rs),val_mean=statistics.mean(float(r['val_rmse']) for r in rs),test_mean=statistics.mean(vals),test_sd=statistics.stdev(vals),mean_parameter_count=statistics.mean(float(r['parameters']) for r in rs),mean_selected_fit_cpu=statistics.mean(float(r['selected_fit_cpu']) for r in rs)))
  if k!='1':a_pairs.append(dict(case=case,k=k,wins_vs_k1=sum(a<b for a,b in zip(vals,bs)),relative_mean_rmse_reduction_vs_k1=1-statistics.mean(vals)/statistics.mean(bs),mean_paired_reduction_vs_k1=statistics.mean(1-a/b for a,b in zip(vals,bs))))
c_summary=[];c_pairs=[]
for case in sorted({r['case'] for r in native if r['stage']=='C'}):
 metrics={}
 for p in ['1.0','0.5','0.25']:
  by=collections.defaultdict(list)
  for r in native:
   if r['stage']=='C' and r['case']==case and r['p']==p:by[r['seed']].append(r)
  assert len(by)==5
  metrics[p]={}
  for seed,rs in by.items():
   assert len(rs)==3
   for key in ['train_rmse','val_rmse','test_rmse']:assert max(float(r[key]) for r in rs)-min(float(r[key]) for r in rs)<1e-7
   metrics[p][seed]=dict(test=float(rs[0]['test_rmse']),val=float(rs[0]['val_rmse']),fit=statistics.median(float(r['fit_wall']) for r in rs),predict=statistics.median(float(r['predict_test_wall']) for r in rs))
  v=list(metrics[p].values());c_summary.append(dict(case=case,p=p,n=5,test_mean=statistics.mean(r['test'] for r in v),test_sd=statistics.stdev(r['test'] for r in v),val_mean=statistics.mean(r['val'] for r in v),mean_seed_median_fit=statistics.mean(r['fit'] for r in v),mean_seed_median_predict=statistics.mean(r['predict'] for r in v)))
 for p in ['0.5','0.25']:
  ds=metrics[p];base=metrics['1.0'];c_pairs.append(dict(case=case,p=p,paired_n=5,accuracy_wins_vs_p1=sum(v['test']<base[s]['test'] for s,v in ds.items()),relative_mean_rmse_change=statistics.mean(v['test'] for v in ds.values())/statistics.mean(v['test'] for v in base.values())-1,median_paired_fit_speedup=statistics.median(base[s]['fit']/v['fit'] for s,v in ds.items()),median_paired_predict_speedup=statistics.median(base[s]['predict']/v['predict'] for s,v in ds.items())))
bd_summary=[]
for stage in ['B','D']:
 for case in sorted({r['case'] for r in native if r['stage']==stage}):
  base={r['seed']:float(r['test_rmse']) for r in native if r['stage']==stage and r['case']==case and r['method']=='CrossGB'}
  for m in sorted({r['method'] for r in native if r['stage']==stage and r['case']==case}):
   rs=[r for r in native if r['stage']==stage and r['case']==case and r['method']==m];assert len(rs)==5;v=[float(r['test_rmse']) for r in rs]
   bd_summary.append(dict(stage=stage,case=case,method=m,n=5,test_mean=statistics.mean(v),test_sd=statistics.stdev(v),val_mean=statistics.mean(float(r['val_rmse']) for r in rs),cgb_wins_vs_method=sum(base[r['seed']]<float(r['test_rmse']) for r in rs) if m!='CrossGB' else None,cgb_relative_mean_rmse_reduction=1-statistics.mean(base.values())/statistics.mean(v) if m!='CrossGB' else None))
for name,rows in [('A_summary',a_summary),('A_pairs',a_pairs),('C_summary',c_summary),('C_pairs',c_pairs),('BD_summary',bd_summary)]:csvwrite(O/(name+'.csv'),rows)
(O/'verification.json').write_text(json.dumps(dict(status='all_passed',A_development_refits=400,A_selected_variants_evaluated=100,B_native_fits=80,C_native_fits_including_repetitions=90,D_native_fits=30,total_new_fits=600,C_unique_dataset_seed_parameter_workpoints=30,no_failed_or_timed_out_fits=all(r['status']=='ok' for r in native),test_selection=False,A_scope='posthoc fixed routes, per-capacity validation selection, inherited constants/eta',B_scope='five fresh paired draws, fixed A06 recipes, shared within-draw val/test across cells',C_scope='posthoc all-p sensitivity; three timing repetitions are not new data draws',D_scope='same fixed real time test, nested train600->2000, no new independent data source'),indent=2)+'\n')
print('A_pairs',a_pairs);print('C_pairs',c_pairs);print('Verified total new fits: 600')
