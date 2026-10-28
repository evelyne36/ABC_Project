#!/usr/bin/env python3
"""Run frozen capacity repetitions or exclusive timing; no parameter selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,datetime,hashlib,json,pathlib,random,signal,subprocess,sys,time

HERE=pathlib.Path(__file__).resolve().parent

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--plan',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--not-before',type=float,default=0);a=p.parse_args()
    plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=False)
    for name,expected in plan['data_sha256'].items():
        if hashlib.sha256((a.root/'fastlane_data'/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Data hash mismatch: '+name)
    write(a.output/'plan.json',plan)
    state=dict(status='waiting_for_batch_start',pid=os.getpid(),kind=plan['kind'],host=plan['host'],completed_tasks=0)
    def status(**values):
        state.update(values);state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(a.output/'status.json',state)
    status()
    if a.not_before>time.time():time.sleep(a.not_before-time.time())
    deadline=time.monotonic()+plan['batch_budget_seconds'];core=min(os.sched_getaffinity(0));rows=[]
    def run(cmd,directory,timeout,info):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Batch elapsed cap reached')
        directory.mkdir(parents=True,exist_ok=True);start=time.monotonic()
        with (directory/'process.log').open('w') as log:
            proc=subprocess.Popen(['taskset','-c',str(core)]+cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(status='running',current=info,child_pid=proc.pid,core=core)
            try:code=proc.wait(timeout=max(.1,min(timeout,remaining)))
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
        row=dict(info,exit_code=code,wall_seconds=time.monotonic()-start)
        rows.append(row);status(child_pid=None,completed_tasks=len(rows));write(a.output/'execution.json',rows)
        return code
    def fit(case,method,params,seed,directory,info,timeout=180,save=False):
        pf=directory/'params.json';write(pf,params)
        cmd=[sys.executable,str(HERE/'worker.py'),'--root',str(a.root),'--data',str(a.root/'fastlane_data'/f'{case}__s{seed}.npz'),
             '--method',method,'--params',str(pf),'--output',str(directory),'--seed',str(seed),'--evaluate-test']
        if save:cmd+=['--save-model']
        return run(cmd,directory,timeout,info)
    try:
        if plan['kind']=='capacity':
            for case,task in plan['cases'].items():
                write(a.output/case/'choice.json',dict(case=case,choices={'CrossGB':{'params':task['params']}},selection='Frozen earlier recipe; no search/selection in followup',source=task['source']))
                for seed in task['seeds']:
                    model=a.output/case/'final/CrossGB'/f's{seed}'
                    code=fit(case,'CrossGB',task['params'],seed,model,dict(case=case,seed=seed,phase='anchor_fit'),save=True)
                    result=model/'result.json'
                    if code or not result.exists() or json.loads(result.read_text()).get('status')!='ok':continue
                    out=a.output/case/'capacity'/f's{seed}'
                    cmd=[sys.executable,str(HERE/'capacity.py'),'--root',str(a.root/'rebuttal_execution_20261025/native/controlled'),
                         '--data',str(a.root/'fastlane_data'/f'{case}__s{seed}.npz'),'--model-dir',str(model),'--output',str(out)]
                    run(cmd,out,300,dict(case=case,seed=seed,phase='capacity'))
        elif plan['kind']=='timing':
            for repeat in range(3):
                for case,task in plan['cases'].items():
                    methods=list(task['methods']);random.Random(170000+repeat).shuffle(methods)
                    for method in methods:
                        fit(case,method,task['methods'][method],98001,a.output/'timing'/case/method/f'r{repeat}',
                            dict(case=case,method=method,repeat=repeat,phase='exclusive_timing'),timeout=120)
        else:raise ValueError('Unknown kind')
        status(status='complete' if all(r['exit_code']==0 for r in rows) else 'complete_with_failures',current=None)
    except Exception as error:
        status(status='stopped',error=type(error).__name__+': '+str(error),child_pid=None)
        raise
    finally:write(a.output/'complete.json',dict(state,execution=rows))

if __name__=='__main__':main()
