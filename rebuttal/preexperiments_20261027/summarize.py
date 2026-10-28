"""Aggregate all bounded preexperiment records without dropping unsuccessful cases."""
from pathlib import Path
import csv,json,statistics,collections,shutil
BASE=Path(__file__).resolve().parents[2];L=BASE/'local_sync';O=Path(__file__).resolve().parent/'results';O.mkdir(exist_ok=True)
def read(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def write(name,rs):
 if not rs:return
 keys=list(dict.fromkeys(k for r in rs for k in r))
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def copy(p,name):shutil.copy2(p,O/name)
scout=L/'optimistic_scout_v1';follow=L/'optimistic_followup_v1';syn=L/'optimistic_synthetic_v1';ext=L/'optimistic_extensions_v1';confirm=L/'optimistic_synthetic_frozen_check_v1'
for folder,names in [(scout,['frozen_plan.json','shortlist_frozen_before_audit.json','supplementary_shortlist_frozen.json','historical_leads.csv','discovery_summary.csv','status.json']),(follow,['frozen_plan.json','status.json']),(syn,['frozen_plan.json','confirmation_frozen_choices.json','status.json']),(ext,['frozen_plan.json','status.json']),(confirm,['frozen_choices.json','status.json'])]:
 for name in names:
  p=folder/name
  if p.exists():copy(p,folder.name+'__'+name)
audit=read(scout/'audit_fits.jsonl')+read(follow/'supplementary_audit.jsonl');write('real_audit_fits.csv',audit)
rr=[]
for cell in sorted({x['cell'] for x in audit}):
 g=[x for x in audit if x['cell']==cell and x['status']=='success'];task=g[0]['task'];r={k:g[0][k] for k in ['cell','dataset','task','mode','n_train','n_validation','n_features']}
 for m in ['CrossGB','XGB','LGB_constant','LGB_linear']:
  z=[x for x in g if x['method']==m];r[m+'_primary']=statistics.mean(x['primary'] for x in z);r[m+'_fit_wall_sec']=statistics.median(x['fit_wall_sec'] for x in z);r[m+'_fit_cpu_sec']=statistics.median(x['fit_cpu_sec'] for x in z)
  if task!='regression':r[m+'_logloss']=statistics.mean(x['logloss'] for x in z)
 for m in ['XGB','LGB_constant','LGB_linear']:
  r['gain_vs_'+m]=r['CrossGB_primary']-r[m+'_primary'] if task=='binary' else (r[m+'_primary']-r['CrossGB_primary'])/r[m+'_primary']
  r['fit_ratio_vs_'+m]=r['CrossGB_fit_wall_sec']/r[m+'_fit_wall_sec']
 rr.append(r)
write('real_audit_summary.csv',rr)
front=read(follow/'real_frontiers.jsonl')+read(ext/'real_lr_frontiers.jsonl');write('real_frontier_fits.csv',front);groups=collections.defaultdict(list)
for x in front:
 if x['status']=='success':groups[(x['cell'],x['method'],x['trees'],x.get('learning_rate',.1))].append(x)
fp=[]
for (cell,m,t,lr),xs in sorted(groups.items()):fp.append(dict(cell=cell,task=xs[0]['task'],method=m,trees=t,learning_rate=lr,primary=statistics.mean(x['primary'] for x in xs),fit_wall_sec=statistics.median(x['fit_wall_sec'] for x in xs),fit_cpu_sec=statistics.median(x['fit_cpu_sec'] for x in xs),repetitions=len(xs)))
write('real_frontier_summary.csv',fp)
rows=read(syn/'fits.jsonl')+read(ext/'synthetic_baseline_extensions.jsonl');write('synthetic_discovery_fits.csv',rows)
g=collections.defaultdict(list)
for x in rows:
 if x['status']=='success':g[(x['case'],x['method'],x['trees'],x['learning_rate'])].append(x)
sp=[]
for (case,m,t,lr),xs in sorted(g.items()):sp.append(dict(case=case,method=m,trees=t,learning_rate=lr,mean_rmse=statistics.mean(x['primary'] for x in xs),sd_rmse=statistics.stdev(x['primary'] for x in xs),median_fit_wall_sec=statistics.median(x['fit_wall_sec'] for x in xs),median_fit_cpu_sec=statistics.median(x['fit_cpu_sec'] for x in xs),data_seeds=len(xs)))
write('synthetic_discovery_frontiers.csv',sp)
cr=read(confirm/'fits.jsonl');write('synthetic_frozen_check_fits.csv',cr);g=collections.defaultdict(list)
for x in cr:
 if x['status']=='success':g[(x['case'],x['method'])].append(x)
cs=[]
for (case,m),xs in sorted(g.items()):cs.append(dict(case=case,method=m,trees=xs[0]['trees'],learning_rate=xs[0]['learning_rate'],mean_rmse=statistics.mean(x['primary'] for x in xs),sd_rmse=statistics.stdev(x['primary'] for x in xs) if len(xs)>1 else None,median_fit_wall_sec=statistics.median(x['fit_wall_sec'] for x in xs),median_fit_cpu_sec=statistics.median(x['fit_cpu_sec'] for x in xs),independent_data_seeds=len(xs)))
write('synthetic_frozen_check_summary.csv',cs)
counts={'discovery_real':len(read(scout/'discovery_fits.jsonl')),'locked_audit_real':len(audit),'real_frontier':len(front),'synthetic_discovery':len(rows),'synthetic_frozen_check':len(cr)}
allrows=read(scout/'discovery_fits.jsonl')+audit+front+rows+cr
status={'fit_attempts':counts,'total':sum(counts.values()),'failed':sum(x['status']!='success' for x in allrows),'formal_real_outer_test_evaluations':0,'audit_cases':len(rr),'real_discovery_cells':len(list(csv.DictReader((scout/'discovery_summary.csv').open()))),'synthetic_cases':len({x['case'] for x in rows}),'warning':'Real shortlist is exploratory selection from local data; audit uses one reserved split, with timing/subsampling repeats, not 3 independent outer splits. Synthetic frozen check uses new data seeds. Training timings exclude preprocessing/imports and HPO.'}
(O/'verification.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print(json.dumps(status,ensure_ascii=False,indent=2))
