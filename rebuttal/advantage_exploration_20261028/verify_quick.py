import pathlib,json,csv,hashlib,collections,tarfile,math,sys
import numpy as np
P=pathlib.Path(__file__).resolve().parents[2];A=P/'rebuttal/advantage_exploration_20261028';R=P/'local_sync/advantage_results';O=A/'quick_report';O.mkdir(exist_ok=True)
manifest=json.loads((A/'dispatch/manifest.json').read_text());rows=[];checks=[]
for lane in manifest['lanes']:
 host=lane['host'];archive=R/(host+'_quick.tar.gz');dest=R/host
 if archive.exists():
  dest.mkdir(exist_ok=True)
  with tarfile.open(archive) as tf:tf.extractall(dest,filter='data')
 base=dest/f'results/advantage_20261028_v1/quick/w{lane["lane"]}'
 plan=json.loads((A/'dispatch'/lane['plan']).read_text());kind=lane['kind']
 entries=json.loads((A/kind/'manifest.json').read_text())['entries'];emap={(e['case'],e['seed']):e for e in entries}
 for j in plan['jobs']:
  f=base/j['relative_output']/'result.json'
  if not f.exists():continue
  r=json.loads(f.read_text());row={k:j[k] for k in ['case','method','candidate','seed']};row.update(kind=kind,host=host,lane=lane['lane'],status=r['status'],val_rmse=None,selection_value=None,fit_wall=None)
  if r['status']=='ok':
   assert r['method']==j['method'] and r['seed']==j['seed'];assert r['requested_params']==j['params'];assert r['worker_sha256']==plan['worker_sha256'];assert r['evaluate_test'] is False and r['test_labels_used'] is False;assert 'test' not in r['metrics']
   e=emap[j['case'],j['seed']];npz=A/kind/(e.get('data_npz') or e['npz']);side=A/kind/(e.get('sidecar') or e['metadata']);meta=json.loads(side.read_text());scale=meta.get('preprocessing',{}).get('y_scale',1.)
   with np.load(npz,allow_pickle=False) as data,np.load(f.parent/'predictions.npz',allow_pickle=False) as pred:
    assert 'test' not in pred.files
    for split in ['train','val']:
     assert np.array_equal(data[split],pred[split+'_indices']);actual=float(np.sqrt(np.mean((pred[split]-data['y_'+split])**2)));assert math.isfinite(actual);assert abs(actual-r[split+'_rmse'])<=1e-8*max(1,actual)
   row.update(val_rmse=r['val_rmse'],selection_value=r['val_rmse']*scale,fit_wall=r['timings']['fit_wall_sec']);checks.append(str(f))
  else:row.update(error=r.get('error'),failure_stage=r.get('failure_stage'))
  rows.append(row)
if rows:
 keys=['case','kind','host','lane','method','candidate','seed','status','val_rmse','selection_value','fit_wall','error','failure_stage']
 with (O/'all_development_fits.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 choices=[]
 for lane in manifest['lanes']:
  case=lane['case']
  for method in ['CrossGB','XGB','LGB_constant','LGB_linear','PL','LinXGBoost']:
   candidates=[]
   for ci in range(4):
    rs=[r for r in rows if r['case']==case and r['method']==method and r['candidate']==ci and r['status']=='ok']
    if len(rs)==2:candidates.append((sum(r['selection_value'] for r in rs)/2,ci,sum(r['fit_wall'] for r in rs)/2))
   if candidates:
    v,ci,t=min(candidates);choices.append(dict(case=case,method=method,candidate=ci,development_rmse=v,fit_wall_diagnostic=t,complete_candidates=len(candidates)))
 (O/'choices_snapshot.json').write_text(json.dumps(choices,indent=2)+'\n')
 print('CASE | CGB | best native-linear | Lin | completed candidates per method')
 for lane in manifest['lanes']:
  c={r['method']:r for r in choices if r['case']==lane['case']}
  val=lambda m:round(c[m]['development_rmse'],6) if m in c else None
  print(lane['case'],val('CrossGB'),min([val(m) for m in ['PL','LGB_linear'] if val(m) is not None],default=None),val('LinXGBoost'),{k:v['complete_candidates'] for k,v in c.items()})
(O/'verification_snapshot.json').write_text(json.dumps(dict(fits_seen=len(rows),statuses=dict(collections.Counter(r['status'] for r in rows)),recomputed_successful_predictions=len(checks),test_used=False),indent=2)+'\n')
print('VERIFIED',len(checks),'STATUSES',dict(collections.Counter(r['status'] for r in rows)))
