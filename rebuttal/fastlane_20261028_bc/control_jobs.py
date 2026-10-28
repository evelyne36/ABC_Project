"""Build SSH-controller requests; this helper never connects or fits."""
import argparse,json,pathlib,shlex
BASE=pathlib.Path(__file__).resolve().parent;WORKSPACE=BASE.parents[1];REMOTE='/home/admin/crossgb_fastlane_20261027'
p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','collect']);p.add_argument('stages',nargs='+');a=p.parse_args();jobs=[]
for stage in a.stages:
 plans=[]
 for f in (BASE/'dispatch'/stage).glob('*.json'):
  if f.name in ['upload_jobs.json','manifest.json']:continue
  plans.append(json.loads(f.read_text()))
 for plan in plans:
  host=plan['host'];dest=f'{REMOTE}/results/formal_bc_20261028_v1/{stage}'
  if a.action=='launch':
   script=f'''import os,pathlib,json,subprocess,datetime
r=pathlib.Path({REMOTE!r});out=pathlib.Path({dest!r})
if out.exists():raise RuntimeError('Immutable output already exists')
env=os.environ.copy();env['PYTHONPATH']=str(r/'.venv/lib/python3.13/site-packages');env['LD_LIBRARY_PATH']=str(r/'usr/lib/x86_64-linux-gnu/lapack')+':'+str(r/'usr/lib/x86_64-linux-gnu/blas')+':/home/admin/anaconda3/lib:'+str(r/'rebuttal_execution_20261025/native/controlled/lib')
cmd=[str(r/'.venv/bin/python'),str(r/'fastlane_bc_v1/batch_runner.py'),'--root',str(r),'--plan',str(r/'fastlane_bc_v1/{stage}_plan.json'),'--output',str(out)]
with (r/'{stage}_v1.log').open('w') as log:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
d={{'pid':p.pid,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':{stage!r}}};(r/'{stage}_v1_launch.json').write_text(json.dumps(d));print(json.dumps(d))'''
   jobs.append({'host':host,'command':'python3 -c '+shlex.quote(script),'save':host+'_'+stage+'_launch.json','timeout':20})
  elif a.action=='status':
   script=f'''import pathlib,json,os,datetime
r=pathlib.Path({dest!r});f=r/'status.json';d=json.loads(f.read_text()) if f.exists() else {{'status':'missing'}}
for k in ['pid','child_pid']:
 try:os.kill(d.get(k),0);d[k+'_alive']=True
 except (ProcessLookupError,TypeError):d[k+'_alive']=False
d['checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(d))'''
   jobs.append({'host':host,'command':'python3 -c '+shlex.quote(script),'save':host+'_'+stage+'_status.json','timeout':20})
  else:
   jobs.append({'host':host,'action':'download','command':"tar -cz --exclude='*/process.log' -C "+REMOTE+' results/formal_bc_20261028_v1/'+stage,'local':str(WORKSPACE/'local_sync'/f'bc_{stage}_{host}.tar.gz'),'timeout':90})
out=BASE/'dispatch'/('_'+a.action+'_'+'_'.join(a.stages)+'.json');out.write_text(json.dumps(jobs,indent=2)+'\n');print(out)
