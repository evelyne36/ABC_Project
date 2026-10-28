from pathlib import Path
import json,hashlib,tarfile,datetime,shlex
from build_dispatch import HERE,ROOT,write,sha
P=HERE.parents[1];RAW=P/'local_sync/last_increment_results';D=HERE/'dispatch/A';inputs=json.loads((HERE/'A_inputs.json').read_text());choices={};uploads=[];launch=[];collect=[];statuses=[]
for host in json.loads((D/'manifest.json').read_text())['hosts'].values():
 p=RAW/(host+'_A_search.tar.gz');dest=RAW/host;dest.mkdir(exist_ok=True)
 with tarfile.open(p) as tf:tf.extractall(dest,filter='data')
 base=dest/'results/last_increment_261028_v1/A_search';assert json.loads((base/'status.json').read_text())['status']=='complete'
 for f in base.glob('*/s*/search.json'):
  x=json.loads(f.read_text());assert len(x['records'])==16 and not x['test_labels_used'];assert x['model_cache_sha256']==sha(f.parent/'models.json');assert x['core_sha256']==sha(HERE/'capacity_core.py')
  assert all(z['lambda1_replay_max_abs']<1e-8 for z in x['records']);key=x['case']+'/s'+str(x['seed']);job=next(j for j in inputs if j['case']==x['case'] and j['seed']==x['seed']);assert x['original_model_sha256']==job['model_sha256'] and x['original_data_sha256']==job['original_data_sha256']
  for k,ch in x['choices'].items():assert ch['val_rmse']==min(z['val_rmse'] for z in x['records'] if str(z['k'])==k)
  choices[key]=dict(search_sha256=sha(f),model_cache_sha256=x['model_cache_sha256'],choices=x['choices'])
assert len(choices)==25,len(choices)
freeze=HERE/'A_freeze.json';assert not freeze.exists();write(freeze,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),choices=choices,posthoc_historical_routes=True,search_test_labels_used=False,local_closed_label_test_passed=True))
E=HERE/'dispatch/A_evaluate';E.mkdir(exist_ok=False)
for host in json.loads((D/'manifest.json').read_text())['hosts'].values():
 plan=json.loads((D/(host+'.json')).read_text());archive=E/(host+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tf:
  tf.add(freeze,arcname='last_increment_v1/A_freeze.json')
  for j in plan['jobs']:
   local=HERE/'inputs/A'/j['case']/('s'+str(j['seed']))/'test_sealed.npz';assert sha(local)==j['test_data_sha256'];tf.add(local,arcname=j['test_data'])
 uploads.append(dict(host=host,action='upload',local=str(archive),remote=ROOT,timeout=120))
 out='results/last_increment_261028_v1/A_evaluate';code="import os,pathlib,json,subprocess\nr=pathlib.Path("+repr(ROOT)+")\nout=r/"+repr(out)+"\nif out.exists():raise RuntimeError('immutable output exists')\nenv=os.environ.copy();env['PYTHONPATH']=str(r/'.venv/lib/python3.13/site-packages')\nwith (r/'last_increment_A_evaluate.log').open('w') as log:p=subprocess.Popen([str(r/'.venv/bin/python'),str(r/'last_increment_v1/A_worker.py'),'--root',str(r),'--plan',str(r/'last_increment_v1/A_plan.json'),'--output',str(out),'--phase','evaluate','--search',str(r/'results/last_increment_261028_v1/A_search'),'--freeze',str(r/'last_increment_v1/A_freeze.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)\nprint(json.dumps(dict(host="+repr(host)+",pid=p.pid)))"
 launch.append(dict(host=host,command='python3 -c '+shlex.quote(code),save=host+'_increment_A_eval_launch.json',timeout=20))
 collect.append(dict(host=host,action='download',command='tar -cz -C '+shlex.quote(ROOT)+' '+out,local=str(RAW/(host+'_A_evaluate.tar.gz')),timeout=120))
 code="import pathlib\nr=pathlib.Path("+repr(ROOT)+")\np=r/"+repr(out+'/status.json')+"\nprint(p.read_text() if p.exists() else (r/'last_increment_A_evaluate.log').read_text()[-2000:])"
 statuses.append(dict(host=host,command='python3 -c '+shlex.quote(code),timeout=20))
for name,jobs in [('upload',uploads),('launch',launch),('collect',collect),('status',statuses)]:write(E/(name+'_jobs.json'),jobs)
print('A choices frozen before evaluation: 25 anchors x4 capacities')
