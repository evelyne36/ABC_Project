#!/usr/bin/env python3
"""Append six pre-existing-small-round hypotheses, with a total cap of30."""
import argparse,concurrent.futures,datetime,json,os,pathlib,shutil,sys
from run_suite import execute,write
HERE=pathlib.Path(__file__).resolve().parent
METHODS=['CrossGB','XGB','LGB_constant','LGB_linear']
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    root=a.root;data=root/'fastlane_data';base=root/'results/formal24_v1';out=root/'results/formal30_class_v1'
    if not (base/'complete.json').exists():raise RuntimeError('Formal24 predecessor not complete')
    if out.exists():raise RuntimeError('New output required')
    out.mkdir(parents=True)
    anchors=json.loads((HERE/'classification_small_round_anchors.json').read_text())
    write(out/'plan.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),basis=anchors,
        parent=str(base),selection='mean val logloss across97001/97002 only',additional_candidates=6,total_candidate_ceiling=30,
        confirmation_seeds=list(range(98001,98006)),test_not_used_in_selection=True))
    cores=sorted(os.sched_getaffinity(0))
    def one(pair):
        i,case=pair;dest=out/case;dest.mkdir();choices={}
        for method in METHODS:
            prior=json.loads((base/case/'candidate_pool.json').read_text())[method][:24]
            scores=[]
            for ci,params in enumerate(prior+anchors['new_candidates']):
                pf=dest/'params'/f'{method}_{ci:02}.json';write(pf,params);vals=[]
                for seed in [97001,97002]:
                    if ci<24:
                        f=base/case/'search'/method/f'c{ci:02}'/f's{seed}'/'result.json'
                        r=json.loads(f.read_text()) if f.exists() else {}
                    else:
                        d=dest/'search'/method/f'c{ci:02}'/f's{seed}'
                        cmd=[sys.executable,str(HERE/'worker.py'),'--root',str(root),'--data',str(data/f'{case}__s{seed}.npz'),
                            '--method',method,'--params',str(pf),'--output',str(d),'--seed',str(seed)]
                        r=execute(cmd,d,60,cores[i])
                    if r.get('status')=='ok' and r.get('val_logloss') is not None:vals.append(r['val_logloss'])
                if len(vals)==2:scores.append((sum(vals)/2,ci))
            if not scores:choices[method]={'status':'no_complete_candidate'};continue
            score,ci=min(scores);choices[method]=dict(candidate=ci,mean_development_val_logloss=score,params=(prior+anchors['new_candidates'])[ci],complete_candidates=len(scores))
        write(dest/'choice.json',dict(case=case,choices=choices,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='development val logloss only, no test metrics read'))
        old=json.loads((base/case/'choice.json').read_text())['choices']
        for method,c in choices.items():
            if 'candidate' not in c:continue
            for seed in range(98001,98006):
                d=dest/'final'/method/f's{seed}'
                src=base/case/'final'/method/f's{seed}'
                if c['params']==old.get(method,{}).get('params') and (src/'result.json').exists():
                    shutil.copytree(src,d);r=json.loads((d/'result.json').read_text());r['reused_identical_frozen_fit_from']=str(src);write(d/'result.json',r)
                else:
                    pf=dest/'params'/f'{method}_{c["candidate"]:02}.json'
                    cmd=[sys.executable,str(HERE/'worker.py'),'--root',str(root),'--data',str(data/f'{case}__s{seed}.npz'),
                        '--method',method,'--params',str(pf),'--output',str(d),'--seed',str(seed),'--evaluate-test']
                    execute(cmd,d,180,cores[i])
        write(dest/'complete.json',dict(status='complete_with_recorded_failures',case=case))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:list(e.map(one,enumerate(['ilpd_numeric','website_phishing'])))
    write(out/'complete.json',dict(status='complete_with_recorded_failures',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
if __name__=='__main__':main()
