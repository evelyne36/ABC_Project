#!/usr/bin/env python3
"""Repeat frozen quick-workpoint fits on a single otherwise idle Linux host."""
import argparse,json,os,pathlib,random,signal,subprocess,sys,time,datetime
HERE=pathlib.Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--recipe',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);recipe=json.loads(a.recipe.read_text());core=min(os.sched_getaffinity(0));rows=[]
    for repeat in range(3):
        for case,choices in recipe.items():
            methods=list(choices);random.Random(150000+repeat).shuffle(methods)
            for method in methods:
                out=a.output/case/method/f'r{repeat}';out.mkdir(parents=True);pf=out/'params.json';pf.write_text(json.dumps(choices[method]))
                cmd=[sys.executable,str(HERE/'worker.py'),'--root',str(a.root),'--data',str(a.root/'fastlane_data'/f'{case}__s98001.npz'),'--method',method,'--params',str(pf),'--output',str(out),'--seed','98001','--evaluate-test']
                with (out/'process.log').open('w') as log:
                    proc=subprocess.Popen(['taskset','-c',str(core)]+cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
                f=out/'result.json';r=json.loads(f.read_text()) if f.exists() else {}
                rows.append(dict(case=case,method=method,repeat=repeat,exit_code=code,status=r.get('status'),val_rmse=r.get('val_rmse'),test_rmse=r.get('test_rmse'),timings=r.get('timings')))
                (a.output/'progress.json').write_text(json.dumps(rows,indent=2))
    (a.output/'complete.json').write_text(json.dumps(dict(rows=rows,core=core,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='3 timing repeats of same frozen train/val/test; not independent statistical repeats; author implementation and API boundaries retained'),indent=2))
if __name__=='__main__':main()
