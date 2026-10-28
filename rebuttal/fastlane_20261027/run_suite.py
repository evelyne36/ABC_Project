#!/usr/bin/env python3
"""Bounded search on two development datasets, then sealed confirmation fits."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import argparse,concurrent.futures,datetime,hashlib,json,pathlib,queue,signal,subprocess,sys,threading,time

HERE=pathlib.Path(__file__).resolve().parent
METHODS=['CrossGB','XGB','LGB_constant','LGB_linear','PL','LinXGBoost']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def grids(synthetic):
    if synthetic:
        base=[(3,1.,1,1.),(2,2.,1,1.),(3,10.,16,.3),(4,1.,64,.1),(3,.1,256,.05),(4,10.,1024,.05)]
    else:base=[(3,1.,64,.1),(2,10.,128,.1),(4,1.,128,.1),(3,.1,256,.05),(5,10.,256,.05),(3,100.,512,.05)]
    # Frozen sequence; prefix is the quick panel, later entries strengthen all methods.
    extended=base+[(d,l,t,lr) for d,l,t,lr in [(2,.1,64,.3),(2,10.,256,.1),(3,10.,256,.1),(4,10.,128,.1),(3,1.,512,.05),(5,1.,128,.1),(2,100.,128,.1),(4,100.,256,.05),(3,.01,128,.1),(5,10.,512,.05),(2,1.,1024,.03),(3,10.,1024,.03),(4,1.,512,.05),(3,100.,256,.1)]]
    extended += [(2,.1,2,1.),(2,1.,4,1.),(3,10.,128,.1),(3,1.,64,.1)] if synthetic else [(2,.1,64,.2),(3,1.,1024,.05),(4,1.,64,.3),(5,10.,300,.1)]
    out={m:[dict(depth=d,ridge=l,trees=t,learning_rate=lr,p=1.) for d,l,t,lr in extended] for m in METHODS[:4]}
    if synthetic:
        # LightGBM first tree is constant: give genuine linear-tree work points.
        for i in [0,1]:out['LGB_linear'][i]['trees']=16;out['LGB_linear'][i]['learning_rate']=.3
    pl=[];lin=[]
    for i,(d,l,t,lr) in enumerate(extended):
        pl.append(dict(num_trees=t,max_depth=d,num_leaves=min(2**d,32),l2_reg=l,max_var=5,
            leaf_type='half_additive' if i%2==0 else 'linear',min_sum_hessians=5. if i%3 else 10.,learning_rate=lr))
        # Author root depth is 1; max_depth=2 is a stump. Include known strong shallow anchor.
        nt=1 if synthetic and i<2 else min(t,50)
        lin.append(dict(n_estimators=nt,max_depth=2 if i%3!=2 else 3,lbda=l,
            min_samples_leaf=10 if i%2==0 else 20,learning_rate=1. if nt==1 else .3))
    out['PL']=pl;out['LinXGBoost']=lin
    if not synthetic:
        for i,(nt,depth,lbda) in enumerate([(5,2,2.),(5,3,20.),(15,2,2.),(15,3,20.),(30,2,2.),(30,2,100.)]):
            out['LinXGBoost'][i]=dict(n_estimators=nt,max_depth=depth,lbda=lbda,min_samples_leaf=10,learning_rate=.3)
        for i in [16,17,18,19]:out['LinXGBoost'][i]['n_estimators']=100 if i%2 else 50
    for i in [22,23]:out['CrossGB'][i]['p']=.5 if i==22 else .25
    return out

def execute(cmd,out,timeout,core):
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    with (out/'process.log').open('w') as log:
        proc=subprocess.Popen(['taskset','-c',str(core)]+cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
    p=out/'result.json'
    result=json.loads(p.read_text()) if p.exists() else {}
    if code!=0 or result.get('status')!='ok':
        result.update(status='timeout' if code==124 else 'failed',process_exit=code,process_wall_sec=time.monotonic()-start)
        write(p,result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--data',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--candidates',type=int,default=6);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--timeout',type=int,default=45);p.add_argument('--confirm-seeds',nargs='+',type=int,default=[98001])
    p.add_argument('--method-budget',type=int,default=0,help='Optional per-method elapsed development cap; zero means candidate-fit caps only')
    p.add_argument('--methods',nargs='+',default=METHODS);p.add_argument('--capacity-cases',nargs='*',default=[])
    p.add_argument('--leafonly-cases',nargs='*',default=[]);a=p.parse_args()
    if not 1<=a.candidates<=24:raise ValueError('At most24 frozen candidates in this runner')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'plan.json').exists():raise RuntimeError('Use a new output directory; existing run is immutable')
    inputs={}
    for case in a.cases:
        for seed in [97001,97002]+a.confirm_seeds:
            f=a.data/f'{case}__s{seed}.npz';inputs[f.name]=sha(f)
    cores=sorted(os.sched_getaffinity(0));assert len(cores)>=a.workers
    order=list(range(a.candidates)) if a.candidates<=6 else [i for i in [0,1,20,21,2,3,22,23]+list(range(4,20)) if i<a.candidates]
    plan=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=a.cases,candidates=a.candidates,
        development_seeds=[97001,97002],confirmation_seeds=a.confirm_seeds,methods=a.methods,candidate_order=order,
        selection='mean source-unit validation RMSE for regression, mean val logloss for classification; both dev successes; never test selection',
        confirmation='frozen params fit on train; val and test separately recorded; no train+val refit',
        cores=cores[:a.workers],workers=a.workers,fit_timeout_sec=a.timeout,final_timeout_sec=180,method_elapsed_cap_sec=a.method_budget,
        worker_sha256=sha(HERE/'worker.py'),data_sha256=inputs,
        capacity_cases=a.capacity_cases,leafonly_cases=a.leafonly_cases,
        real_confirmation_caveat='real repeats share fixed heldout; synthetic seeds independently generated')
    write(a.output/'plan.json',plan)
    core_pool=queue.Queue()
    for core in cores[:a.workers]:core_pool.put(core)
    thread_state=threading.local()
    def case_job(pair):
        ix,case=pair
        if not hasattr(thread_state,'core'):thread_state.core=core_pool.get()
        core=thread_state.core;dest=a.output/case;dest.mkdir(exist_ok=True)
        grid=grids(case.startswith('S'));write(dest/'candidate_pool.json',grid)
        choices={};all_records=[]
        for method in a.methods:
            scores=[];method_start=time.monotonic();unstarted=[]
            for order_index,ci in enumerate(order):
                params=grid[method][ci]
                if a.method_budget and time.monotonic()-method_start>=a.method_budget:
                    unstarted=order[order_index:];break
                pf=dest/'params'/f'{method}_{ci:02}.json';write(pf,params);rs=[]
                for seed in [97001,97002]:
                    out=dest/'search'/method/f'c{ci:02}'/f's{seed}'
                    cmd=[sys.executable,str(HERE/'worker.py'),'--root',str(a.root),'--data',str(a.data/f'{case}__s{seed}.npz'),
                        '--method',method,'--params',str(pf),'--output',str(out),'--seed',str(seed)]
                    remaining=a.method_budget-(time.monotonic()-method_start) if a.method_budget else a.timeout
                    if remaining<=0:r={'status':'budget_exhausted'};write(out/'result.json',r)
                    else:r=execute(cmd,out,max(1,min(a.timeout,remaining)),core)
                    meta=json.loads((a.data/f'{case}__s{seed}.json').read_text())
                    scale=meta.get('preprocessing',{}).get('y_scale',1.)
                    classification=meta.get('task_type','regression')!='regression'
                    r['selection_value']=(r.get('val_logloss') if classification else r.get('val_rmse',0.)*scale) if r.get('status')=='ok' else None
                    r['selection_metric']='logloss' if classification else 'rmse_source_units'
                    rs.append(r)
                    all_records.append(dict(case=case,method=method,candidate=ci,seed=seed,status=r['status'],val_rmse=r.get('val_rmse'),val_logloss=r.get('val_logloss'),selection_value=r['selection_value'],selection_metric=r['selection_metric']))
                    write(dest/'progress.json',dict(phase='search',attempts=len(all_records),last=all_records[-1]))
                if all(r.get('status')=='ok' and r['selection_value'] is not None for r in rs):scores.append((sum(r['selection_value'] for r in rs)/2,ci))
            if scores:
                best,ci=min(scores);choices[method]=dict(candidate=ci,mean_development_primary=best,selection_metric=rs[-1]['selection_metric'],params=grid[method][ci],unstarted_candidates=unstarted,complete_candidates=len(scores),development_elapsed_sec=time.monotonic()-method_start)
            else:choices[method]=dict(status='no_complete_candidate')
        # Atomic freeze of ALL method selections before any confirmation test is opened.
        write(dest/'choice.json',dict(case=case,choices=choices,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),search_records=all_records))
        for seed in a.confirm_seeds:
            for method in a.methods:
                choice=choices[method]
                if 'candidate' not in choice:continue
                out=dest/'final'/method/f's{seed}';pf=dest/'params'/f'{method}_{choice["candidate"]:02}.json'
                cmd=[sys.executable,str(HERE/'worker.py'),'--root',str(a.root),'--data',str(a.data/f'{case}__s{seed}.npz'),
                    '--method',method,'--params',str(pf),'--output',str(out),'--seed',str(seed),'--evaluate-test']
                if method=='CrossGB':cmd+=['--save-model']
                r=execute(cmd,out,180,core)
                if method=='CrossGB' and r.get('status')=='ok':
                    if case in a.leafonly_cases:
                        leafout=dest/'final/LeafOnly'/f's{seed}'
                        leafcmd=[sys.executable,str(HERE/'worker.py'),'--root',str(a.root),'--data',str(a.data/f'{case}__s{seed}.npz'),
                            '--method','LeafOnly','--params',str(pf),'--output',str(leafout),'--seed',str(seed),'--evaluate-test']
                        execute(leafcmd,leafout,180,core)
                    if case in a.capacity_cases:
                        capout=dest/'capacity'/f's{seed}'
                        cmd=[sys.executable,str(HERE/'capacity.py'),'--root',str(a.root/'rebuttal_execution_20261025/native/controlled'),
                            '--data',str(a.data/f'{case}__s{seed}.npz'),'--model-dir',str(out),'--output',str(capout)]
                        # capacity has a different result schema; preserve raw exit/diagnostics.
                        capout.mkdir(parents=True,exist_ok=True)
                        t=time.monotonic()
                        with (capout/'process.log').open('w') as log:
                            cp=subprocess.Popen(['taskset','-c',str(core)]+cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            try:cc=cp.wait(timeout=300)
                            except subprocess.TimeoutExpired:os.killpg(cp.pid,signal.SIGKILL);cp.wait();cc=124
                        write(capout/'execution.json',dict(exit_code=cc,wall_sec=time.monotonic()-t))
        write(dest/'complete.json',dict(case=case,status='complete_with_recorded_failures',search_attempts=len(all_records),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print(json.dumps(dict(case=case,status='complete',choices=choices)),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:list(ex.map(case_job,enumerate(a.cases)))
    write(a.output/'complete.json',dict(cases=a.cases,status='complete_with_recorded_failures',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
if __name__=='__main__':main()
