"""Package frozen independent fit jobs; no model or dataset selection."""
import pathlib,json,hashlib,tarfile,shlex,datetime
HERE=pathlib.Path(__file__).resolve().parent
HOSTS=['cpu1_pool.zjuici.com','cpu2_pool.zjuici.com','cpu4_pool.zjuici.com','cpu5_pool.zjuici.com','cpu6_pool.zjuici.com','cpu_Calculator','TabularData_Processor']
ROOT='/home/admin/crossgb_fastlane_20261027'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def package(stage,jobs,datafiles,core_slot=2,wall_budget_seconds=1800,host_assignment=None):
 d=HERE/'dispatch'/stage;d.mkdir(parents=True,exist_ok=False);uploads=[];launches=[];statuses=[];collects=[];lanes=[]
 for hi,host in enumerate(HOSTS):
  selected=[j for ix,j in enumerate(jobs) if (host_assignment[j['case']]==host if host_assignment else ix%7==hi)]
  if not selected:continue
  plan=dict(stage=stage,host=host,core_slot=core_slot,worker_sha256=sha(HERE/'worker.py'),wall_budget_seconds=wall_budget_seconds,data_sha256={j['data']:sha(datafiles[j['data']]) for j in selected},jobs=selected)
  pf=d/(host+'.json');write(pf,plan);archive=d/(host+'.tar.gz')
  with tarfile.open(archive,'w:gz') as t:
   for remote in plan['data_sha256']:t.add(datafiles[remote],arcname=remote)
   for f in ['runner.py','worker.py']:t.add(HERE/f,arcname='advantage_v1/'+f)
   t.add(pf,arcname='advantage_v1/'+stage+'_plan.json')
  uploads.append(dict(host=host,action='upload',local=str(archive),remote=ROOT,timeout=120))
  out='results/advantage_20261028_v1/'+stage
  code="import os,pathlib,json,subprocess,datetime\nr=pathlib.Path("+repr(ROOT)+")\nout=r/"+repr(out)+"\nif out.exists():raise RuntimeError('Immutable output exists')\nenv=os.environ.copy();env['PYTHONPATH']=str(r/'.venv/lib/python3.13/site-packages');env['LD_LIBRARY_PATH']=':'.join([str(r/'usr/lib/x86_64-linux-gnu/lapack'),str(r/'usr/lib/x86_64-linux-gnu/blas'),'/home/admin/anaconda3/lib',str(r/'rebuttal_execution_20261025/native/controlled/lib')])\n"
  code+="with (r/"+repr('advantage_'+stage+'.log')+").open('w') as log:p=subprocess.Popen([str(r/'.venv/bin/python'),str(r/'advantage_v1/runner.py'),'--root',str(r),'--plan',str(r/"+repr('advantage_v1/'+stage+'_plan.json')+"),'--output',str(out)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)\nprint(json.dumps(dict(host="+repr(host)+",pid=p.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))"
  launches.append(dict(host=host,command='python3 -c '+shlex.quote(code),save=host+'_advantage_'+stage+'_launch.json',timeout=25))
  code="import pathlib,json\np=pathlib.Path("+repr(ROOT+'/'+out+'/status.json')+")\nprint(p.read_text() if p.exists() else json.dumps(dict(status='missing')))"
  statuses.append(dict(host=host,command='python3 -c '+shlex.quote(code),save=host+'_advantage_'+stage+'_status.json',timeout=20))
  local=HERE.parents[1]/'local_sync/advantage_results';local.mkdir(exist_ok=True)
  collects.append(dict(host=host,action='download',command='tar -cz -C '+shlex.quote(ROOT)+' '+shlex.quote(out),local=str(local/(host+'_'+stage+'.tar.gz')),timeout=90))
  lanes.append(dict(host=host,plan=pf.name,jobs=len(selected)))
 for name,x in [('upload',uploads),('launch',launches),('status',statuses),('collect',collects)]:write(d/(name+'_jobs.json'),x)
 write(d/'manifest.json',dict(stage=stage,lanes=lanes,total_jobs=len(jobs),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 return d
if __name__=='__main__':
 pools=json.loads((HERE/'frozen_candidate_pools.json').read_text());jobs=[];datafiles={}
 finalists=[('synthetic','A06_two_sparse_modules'),('real','gas_turbine_nox')]
 write(HERE/'finalist_freeze.json',dict(selected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),finalists=finalists,basis='Development-only: A06 distinct additive sparse mechanism, positive precision/cost screen. Gas is near-tie to PL and potential real accuracy-cost tradeoff, not a declared winner. A04 and n/d variants retained as screening; all other real tasks retained.',quick_report_sha256=sha(HERE/'quick_report/choices_snapshot.json'),test_seen=False,candidate_pool_sha256=sha(HERE/'frozen_candidate_pools.json')))
 for kind,case in finalists:
  entries=[e for e in json.loads((HERE/kind/'manifest.json').read_text())['entries'] if e['case']==case]
  for ci in range(4,20):
   for method,configs in pools[kind].items():
    for e in entries:
     f=HERE/kind/(e.get('data_npz') or e['npz']);remote='advantage_inputs/'+kind+'/'+f.name;datafiles[remote]=f;seed=e['seed'];jid=f'{case}/{method}/c{ci:02}/s{seed}'
     jobs.append(dict(id=jid,case=case,kind=kind,method=method,candidate=ci,seed=seed,params=configs[ci],data=remote,relative_output=jid,evaluate_test=False,timeout_seconds=180))
 d=package('hpo',jobs,datafiles);print(json.dumps(dict(dispatch=str(d),fits=len(jobs),hosts=7)))
