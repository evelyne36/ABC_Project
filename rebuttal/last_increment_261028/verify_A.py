from pathlib import Path
import json,csv,tarfile,hashlib,math,collections
import numpy as np
import capacity_core as core
I=Path(__file__).resolve().parent;ROOT=I.parents[1];RAW=ROOT/'local_sync/last_increment_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=json.loads((I/'A_inputs.json').read_text());freeze=I/'A_freeze.json';frozen=json.loads(freeze.read_text());rows=[];max_replay=0
for host in json.loads((I/'dispatch/A/manifest.json').read_text())['hosts'].values():
 p=RAW/(host+'_A_evaluate.tar.gz');dest=RAW/host
 with tarfile.open(p) as tf:tf.extractall(dest,filter='data')
 base=dest/'results/last_increment_261028_v1/A_evaluate';assert json.loads((base/'status.json').read_text())['status']=='complete'
 for result in base.glob('*/s*/result.json'):
  r=json.loads(result.read_text());assert r['status']=='ok' and r['freeze_sha256']==sha(freeze);job=next(j for j in inputs if j['case']==r['case'] and j['seed']==r['seed']);local=I/'inputs/A'/r['case']/('s'+str(r['seed']));cache=dest/'results/last_increment_261028_v1/A_search'/r['case']/('s'+str(r['seed']));search=json.loads((cache/'search.json').read_text());models=json.loads((cache/'models.json').read_text());assert sha(cache/'models.json')==search['model_cache_sha256'];assert r['choices']==search['choices'];assert len(search['records'])==16
  with np.load(local/'search.npz',allow_pickle=False) as a,np.load(local/'test_sealed.npz',allow_pickle=False) as t,np.load(result.parent/'predictions.npz',allow_pickle=False) as pred:
   X=a['X'].astype(np.float32).astype(np.float64);indices={s:a[s] for s in ['train','val','test']};target={s:a['y_'+s] for s in ['train','val']};target['test']=t['y_test'];raw=json.loads((local/'model.json').read_text())['learner'];trees=raw['gradient_booster']['model']['trees'];base_score=core.scalar(raw['learner_model_param']['base_score']);routes={s:core.crossgb_leaf_routes(X[idx],trees) for s,idx in indices.items()}
   for k,model in models.items():
    row=dict(case=r['case'],seed=r['seed'],k=k,multiplier=r['choices'][k]['multiplier'],ridge=r['choices'][k]['ridge'])
    for s in ['train','val','test']:
     assert np.array_equal(pred[s+'_indices'],indices[s]);value=pred[f'k{k}__{s}'];expected=core.predict_frozen(X[indices[s]],routes[s],model,base_score);diff=float(np.max(np.abs(value-expected)));max_replay=max(max_replay,diff);assert diff<1e-8
     score=float(np.sqrt(np.mean((value-target[s])**2)));assert math.isfinite(score) and abs(score-r['metrics'][k][s])<1e-8;row[s+'_rmse']=score*r['scale']
    assert abs(r['metrics'][k]['val']-search['choices'][k]['val_rmse'])<1e-8;row.update(parameters=core.parameter_counts(model)['total_scalar_parameters_including_base'],selected_fit_cpu=r['choices'][k]['fit_cpu'],selected_fit_wall=r['choices'][k]['fit_wall']);rows.append(row)
assert len(rows)==100,len(rows)
out=I/'reports';out.mkdir(exist_ok=True)
with (out/'A_selected.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'A_verification.json').write_text(json.dumps(dict(status='passed',anchors=25,development_local_refits=400,selected_variants=100,all_split_metrics_recomputed=300,all_split_model_replays=300,max_abs_replay_difference=max_replay,search_did_not_load_test_labels=True,global_selection_frozen_before_evaluation=True,scope='posthoc historical routes; not independent confirmation or jointly retrained k>1'),indent=2)+'\n')
print('A verified',len(rows),'selected variants; max prediction replay difference',max_replay)
for case in sorted({r['case'] for r in rows}):
 print(case,[(k,round(float(np.mean([r['test_rmse'] for r in rows if r['case']==case and r['k']==k])),5),[r['multiplier'] for r in rows if r['case']==case and r['k']==k]) for k in ['1','2','3','all']])
