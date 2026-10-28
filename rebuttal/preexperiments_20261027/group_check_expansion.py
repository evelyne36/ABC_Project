"""Feature-group stress check for the two duplicate-rich added classification datasets."""
from pathlib import Path
import argparse,json,csv,time,statistics,hashlib
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-project',type=Path,required=True);p.add_argument('--xgboost-python-package',type=Path,required=True);p.add_argument('--lightgbm-package',type=Path,required=True);p.add_argument('--previous-results',type=Path,required=True);p.add_argument('--expansion',type=Path,required=True);a=p.parse_args();D=a.data_project.resolve();E=a.expansion.resolve();O=E/'feature_group_check';O.mkdir(exist_ok=False)
import model_runtime as rt
from sklearn.model_selection import GroupShuffleSplit
rt.initialize(a.xgboost_python_package,a.lightgbm_package,O/'mpl-cache');np,pd=rt.np,rt.pd
base=json.loads((a.previous_results/'optimistic_scout_v1__frozen_plan.json').read_text());original=json.loads((E/'frozen_plan.json').read_text());names=['maternal_health_risk.csv','website_phishing.csv'];items=[x for x in original['cells'] if x['dataset'] in names];methods=['XGB','CrossGB','LGB_constant','LGB_linear']
(O/'frozen_plan.json').write_text(json.dumps({'datasets':names,'seeds':[90101,90102,90103,90104,90105],'change':'Group by exact raw model-feature tuple within preserved outer-training pool; same group only on one side. Existing parameter choices unchanged.','purpose':'Stress-test generalization to unseen feature combinations; a different estimand from IID row splits, not proof that duplicates invalidate IID evaluation.','formal_outer_test_evaluations':0,'fit_attempts_expected':40},ensure_ascii=False,indent=2)+'\n')
rows=[];diagnostics=[]
with rt.threadpool_limits(limits=1):
 for item in items:
  inv=next(x for x in base['inventory'] if x['dataset']==item['dataset']);X,y=rt.load(D/inv['path'],item['task']);outer,test,_,_=rt.split_data(X,y,item['task']);modelX=X.select_dtypes(include=[np.number]) if item['mode']=='drop' else X
  groups=pd.util.hash_pandas_object(modelX,index=False).to_numpy()
  diagnostics.append({'dataset':item['dataset'],'rows':len(X),'unique_feature_groups':len(np.unique(groups)),'duplicated_feature_rows':len(X)-len(np.unique(groups)),'source_sha256':inv['sha256']})
  for seed in [90101,90102,90103,90104,90105]:
   ii,jj=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(outer,y.iloc[outer],groups=groups[outer]));it,iv=outer[ii],outer[jj]
   assert not set(groups[it])&set(groups[iv]) and not(set(it)|set(iv))&set(test)
   A,B,yt,yv,ym,ys,labels=rt.prep(X,y,item['task'],it,iv,item['mode'])
   for method in methods:
    ci=item[method+'_frozen_config'];r={'dataset':item['dataset'],'cell':item['cell'],'task':item['task'],'seed':seed,'method':method,'config':ci,'n_train':len(it),'n_validation':len(iv),'train_groups':len(np.unique(groups[it])),'validation_groups':len(np.unique(groups[iv])),'cross_split_group_overlap':0,'status':'success'}
    try:r.update(rt.fit_one(method,item['task'],base['configs'][ci],seed,A,B,yt,yv,ym,ys))
    except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
    rows.append(r)
    with (O/'fits.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
  print('GROUP CHECK',item['cell'],'done',flush=True)
summary=[]
for item in items:
 rs=[x for x in rows if x['dataset']==item['dataset'] and x['status']=='success'];r={'dataset':item['dataset'],'cell':item['cell'],'task':item['task'],'repeats':5}
 for m in methods:
  z=[x for x in rs if x['method']==m];r[m+'_primary_mean']=statistics.mean(x['primary'] for x in z);r[m+'_primary_sd']=statistics.stdev(x['primary'] for x in z);r[m+'_fit_wall_median']=statistics.median(x['fit_wall_sec'] for x in z)
 for m in ['XGB','LGB_constant','LGB_linear']:
  gains=[]
  for seed in [90101,90102,90103,90104,90105]:
   c=next(x['primary'] for x in rs if x['seed']==seed and x['method']=='CrossGB');b=next(x['primary'] for x in rs if x['seed']==seed and x['method']==m);gains.append((b-c)/b)
  r['mean_paired_gain_vs_'+m]=statistics.mean(gains);r['wins_vs_'+m]=sum(v>0 for v in gains);r['gain_from_mean_metrics_vs_'+m]=(r[m+'_primary_mean']-r['CrossGB_primary_mean'])/r[m+'_primary_mean']
 summary.append(r)
with (O/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(O/'diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n');(O/'verification.json').write_text(json.dumps({'fit_attempts':len(rows),'success':sum(x['status']=='success' for x in rows),'overlapping_feature_groups':0,'formal_outer_test_evaluations':0},indent=2)+'\n')
for r in summary:print(r['dataset'],r['CrossGB_primary_mean'],'gainX',r['mean_paired_gain_vs_XGB'],'gainL',r['mean_paired_gain_vs_LGB_linear'],'wins',r['wins_vs_XGB'],r['wins_vs_LGB_linear'],flush=True)
