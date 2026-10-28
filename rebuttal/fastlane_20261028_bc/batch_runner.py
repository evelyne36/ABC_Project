#!/usr/bin/env python3
"""Execute a frozen list of individual fits with no model selection."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse,datetime,hashlib,json,pathlib,signal,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--plan',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=False)
    if sha(HERE/'worker.py')!=plan['worker_sha256']:raise RuntimeError('Worker fingerprint mismatch')
    for f,h in plan['data_sha256'].items():
        if sha(a.root/f)!=h:raise RuntimeError('Data fingerprint mismatch: '+f)
    write(a.output/'plan.json',plan);core=min(os.sched_getaffinity(0));deadline=time.monotonic()+plan['wall_budget_seconds'];rows=[]
    state=dict(status='running',pid=os.getpid(),core=core,stage=plan['stage'],completed=0,total=len(plan['jobs']))
    def status(**kw):
        state.update(kw);state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(a.output/'status.json',state)
    status()
    for job in plan['jobs']:
        out=a.output/job['relative_output'];out.mkdir(parents=True,exist_ok=False);pf=out/'params.json';write(pf,job['params'])
        if time.monotonic()>=deadline:
            write(out/'result.json',dict(status='budget_exhausted',job_id=job['id']));rows.append(dict(job_id=job['id'],status='budget_exhausted'));continue
        cmd=[sys.executable,str(HERE/'worker.py'),'--root',str(a.root),'--data',str(a.root/job['data']),
             '--method',job['method'],'--params',str(pf),'--output',str(out),'--seed',str(job['seed'])]
        if job.get('evaluate_test'):cmd+=['--evaluate-test']
        if job.get('save_model'):cmd+=['--save-model']
        start=time.monotonic()
        with (out/'process.log').open('w') as log:
            child=subprocess.Popen(['taskset','-c',str(core)]+cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(current=job['id'],child_pid=child.pid)
            try:code=child.wait(timeout=max(.1,min(job['timeout_seconds'],deadline-time.monotonic())))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
        f=out/'result.json';result=json.loads(f.read_text()) if f.exists() else {}
        if code or result.get('status')!='ok':
            result.update(status='timeout' if code==124 else 'failed',exit_code=code,job_id=job['id']);write(f,result)
        rows.append(dict(job_id=job['id'],status=result['status'],exit_code=code,elapsed_seconds=time.monotonic()-start))
        write(a.output/'execution.json',rows);status(completed=len(rows),child_pid=None)
    status(status='complete' if all(r['status']=='ok' for r in rows) else 'complete_with_recorded_failures',current=None,completed=len(rows))
    write(a.output/'complete.json',dict(state,execution=rows))
if __name__=='__main__':main()
