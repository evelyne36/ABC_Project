#!/usr/bin/env python3
"""Nested 5-outer/4-inner evaluation for the LightGBM neighbours; plan-only by default.

Line-for-line copy of `scripts/run_nested.py` (same folds, same 300-candidate
search, same selection rule "mean of all four inner RMSE", same output layout)
with exactly four differences:

  1. `--method` is one of lgb_constant / lgb_linear;
  2. each fit is launched through `scripts/run_task_lgb.py`;
  3. the plan's adapter digest hashes `lgb_ext/adapter.py` and the runner
     itself instead of `baselines/estimators.py` + `baselines/` fingerprint,
     because those hashed files are deliberately left untouched so the
     original four-method panels keep their resume identity;
  4. provenance is the installed `lightgbm` distribution, not a pinned author
     checkout.

Keep this file's protocol logic in sync with `scripts/run_nested.py`; the only
intended divergence is the four points above.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.protocol import load_protocol

METHODS=["lgb_constant","lgb_linear"]
ADAPTER_PATH=ROOT/"lgb_ext"/"adapter.py"
RUNNER_PATH=ROOT/"scripts"/"run_task_lgb.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_input(path, **arrays):
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                raise RuntimeError("Existing fit input differs from frozen protocol: "+str(path))
    else:
        np.savez_compressed(path,**arrays)


def lightgbm_source_record(method):
    import lightgbm
    return {"method":method,"implementation":"official_lightgbm_pip","distribution":"lightgbm",
            "version":lightgbm.__version__,"library_file":str(lightgbm.__file__),
            "library_sha256":digest(lightgbm.__file__),"linear_tree":method=="lgb_linear",
            "note":("LightGBM linear_tree is documented by the GBDT-PL authors as an approximate "
                    "implementation of GBDT-PL; it is not the PL-Trees code.")}


def execute_fit(method, source, params, directory, seed, standardize, timeout, resume):
    report=directory/"result.json"
    if resume and report.is_file():
        cached=json.loads(report.read_text())
        if (cached.get("status")=="success" and cached.get("input_sha256")==digest(source)
            and cached.get("requested_params")==params and cached.get("seed")==seed
            and cached.get("metadata",{}).get("standardize_training_only")==standardize):
            return {**cached,"cached":True}
    attempt=directory
    number=0
    while attempt.exists():
        number+=1;attempt=directory.with_name(directory.name+f"_retry{number}")
    command=[sys.executable,str(RUNNER_PATH),"--method",method,"--input",str(source),
             "--output",str(attempt),"--params",json.dumps(params),"--seed",str(seed)]
    if standardize:
        command.append("--standardize")
    start=time.perf_counter()
    log_path=attempt.with_suffix(".log")
    with log_path.open("w") as log:
        try:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            if completed.returncode==0:
                return {**json.loads((attempt/"result.json").read_text()),"cached":False}
            error="process_exit_"+str(completed.returncode)
        except subprocess.TimeoutExpired:
            error="fit_timeout"
    return {"status":"failed","error":error,"job_wall_sec":time.perf_counter()-start,"log":str(log_path),"cached":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",choices=METHODS,required=True)
    p.add_argument("--data",type=Path,required=True,help="Directory prepared by prepare_protocol.py")
    p.add_argument("--candidate-file",type=Path,required=True,help="Frozen JSON list of LGBMRegressor parameter dictionaries")
    p.add_argument("--outer-fold",type=int,default=0,help="1..5, or 0 for all five")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=1,help="Concurrent independent fit processes")
    p.add_argument("--fit-timeout",type=int,default=600)
    p.add_argument("--seed",type=int)
    p.add_argument("--standardize",action="store_true")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--resume",action="store_true",help="Reuse matching successful tasks; preserve failed attempts")
    p.add_argument("--label",default="followup",help="Use integration_check for small debugging runs")
    args=p.parse_args()
    candidates=json.loads(args.candidate_file.read_text())
    if not isinstance(candidates,list) or not candidates or not all(isinstance(r,dict) for r in candidates):
        p.error("Candidate file must be a nonempty list of parameter objects")
    if args.workers<1 or args.workers>32 or args.fit_timeout<1 or args.outer_fold not in range(6):
        p.error("Use 1..32 workers, a positive timeout, and outer-fold 0..5")
    X,y,folds,manifest=load_protocol(args.data)
    if args.outer_fold:
        folds=[f for f in folds if f["outer_fold"]==args.outer_fold]
    seed=args.seed if args.seed is not None else (271828 if "data_seed" in manifest["source"] else 42)
    budget=manifest["source"]["historical_cgb_candidates"]
    plan={"method":args.method,"dataset":manifest["dataset"],"outer_folds":[f["outer_fold"] for f in folds],
          "candidate_count":len(candidates),"inner_folds":4,"planned_fits":sum(len(candidates)*len(f["inner"])+1 for f in folds),
          "crossgb_fit_calls":0,"budget_matches_archived_cgb_candidate_count":len(candidates)==budget,
          "archived_cgb_candidate_count":budget,"seed":seed,"label":args.label,"execute":args.execute,
          "data_checksum":manifest["data_npz_sha256"],"candidate_file_sha256":digest(args.candidate_file),
          "adapter_sha256":digest(ADAPTER_PATH),"task_runner_sha256":digest(RUNNER_PATH),
          "harness":"scripts/run_nested_lgb.py (copy of run_nested.py protocol; method dispatch only)",
          "standardize":args.standardize}
    plan["source"]=lightgbm_source_record(args.method)
    if not args.execute:
        print(json.dumps(plan,indent=2));return 0
    if args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; select a new directory or explicitly use --resume")
    args.output.mkdir(parents=True,exist_ok=args.resume)
    plan_path=args.output/"plan.json"
    identity={k:v for k,v in plan.items() if k not in {"outer_folds","execute"}}
    if plan_path.exists():
        previous=json.loads(plan_path.read_text())
        if {k:v for k,v in previous.items() if k not in {"outer_folds","execute"}}!=identity:
            raise RuntimeError("Resume plan identity changed; use a new result directory")
    else:
        plan_path.write_text(json.dumps(plan,indent=2))
    overall=[]
    for fold in folds:
        out=args.output/f"outer{fold['outer_fold']}";out.mkdir(exist_ok=args.resume)
        # Prepare each fold input once; all candidate configurations share it.
        sources=[]
        for j,inner in enumerate(fold["inner"],1):
            path=out/f"inner{j}.npz"
            frozen_input(path,X_train=X[inner["train"]],y_train=y[inner["train"]],X_predict=X[inner["validation"]],y_evaluate=y[inner["validation"]])
            sources.append(path)
        jobs=[(i,j,params,path) for i,params in enumerate(candidates,1) for j,path in enumerate(sources,1)]
        def run(job):
            i,j,params,path=job
            result=execute_fit(args.method,path,params,out/f"candidate{i:04d}_inner{j}",seed,args.standardize,args.fit_timeout,args.resume)
            return i,j,result
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results=list(pool.map(run,jobs))
        scores=[]
        for i,params in enumerate(candidates,1):
            rows=[r for c,_,r in results if c==i]
            success=all(r["status"]=="success" for r in rows)
            scores.append({"candidate_index":i,"status":"success" if success else "failed", "mean_inner_rmse":float(np.mean([r["rmse"] for r in rows])) if success else None,
                           "params":params,"successful_folds":sum(r["status"]=="success" for r in rows),
                           "failures":[r for r in rows if r["status"]!="success"]})
        (out/"candidate_scores.json").write_text(json.dumps(scores,indent=2,allow_nan=False))
        usable=[r for r in scores if r["status"]=="success"]
        if not usable:
            (out/"summary.json").write_text(json.dumps({"status":"failed","reason":"no candidate completed every inner fold"},indent=2))
            raise RuntimeError("No valid candidate; inspect candidate_scores.json and fit logs")
        selected=min(usable,key=lambda r:(r["mean_inner_rmse"],r["candidate_index"]))
        final_input=out/"outer_refit.npz"
        frozen_input(final_input,X_train=X[fold["train"]],y_train=y[fold["train"]],X_predict=X[fold["test"]],y_evaluate=y[fold["test"]])
        final=execute_fit(args.method,final_input,selected["params"],out/"selected_refit",seed,args.standardize,args.fit_timeout,args.resume)
        if final["status"]!="success":
            raise RuntimeError("Selected outer refit failed: "+json.dumps(final))
        summary={**plan,"outer_fold":fold["outer_fold"],"status":"success","test_rmse":final["rmse"],
                 "selected_candidate_index":selected["candidate_index"],"selected_params":selected["params"],
                 "selected_mean_inner_rmse":selected["mean_inner_rmse"],"selection":"mean of all four inner RMSE; outer test evaluated only after selection",
                 "new_fit_processes":sum(not r["cached"] for _,_,r in results)+int(not final["cached"]),
                 "failed_candidates":len(scores)-len(usable),"metadata":final["metadata"]}
        (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
        overall.append(summary)
        print(json.dumps({"outer_fold":fold["outer_fold"],"status":"success","failed_candidates":summary["failed_candidates"],"label":args.label}),flush=True)
    completed=[json.loads(path.read_text()) for path in sorted(args.output.glob("outer*/summary.json"))]
    completed=[r for r in completed if r.get("status")=="success"]
    (args.output/"summary.json").write_text(json.dumps({"plan":plan,"completed_outer_folds":len(completed),
        "completed_indices":[r["outer_fold"] for r in completed],"mean_test_rmse":float(np.mean([r["test_rmse"] for r in completed])),"crossgb_fit_calls":0},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
