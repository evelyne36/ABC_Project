#!/usr/bin/env python3
"""Run four isolated tiny author-model fits; these are integration checks only."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PARAMS={
 "linxgboost":{"n_estimators":5,"max_depth":2,"lbda":0.1,"min_samples_leaf":8},
 "pilot":{"max_depth":3,"max_model_depth":8,"min_sample_leaf":8},
 "tao":{"depth":4,"max_iter":3,"C_dnode":10},
 "gbdt_pl":{"num_trees":12,"num_leaves":4,"max_var":3,"learning_rate":0.2,"min_sum_hessians":4,"num_bins":31},
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,help="Optional prepared real dataset; use its first historical outer fold")
    args=parser.parse_args()
    directory=ROOT/"reports"/("smoke_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    directory.mkdir()
    source=directory/"fixture.npz"
    if args.data:
        from baselines.protocol import load_protocol
        X,y,folds,manifest=load_protocol(args.data)
        train,test=folds[0]["train"],folds[0]["test"]
        np.savez_compressed(source,X_train=X[train],y_train=y[train],X_predict=X[test],y_evaluate=y[test])
        expected_rows=len(test);fixture_name=manifest["dataset"]+"_historical_outer1"
    else:
        rng=np.random.default_rng(20261025)
        X=rng.uniform(-1,1,size=(144,3))
        y=2*X[:,0]-X[:,1]+0.5*X[:,2]+4*(X[:,0]>0.15)
        np.savez_compressed(source,X_train=X[:112],y_train=y[:112],X_predict=X[112:],y_evaluate=y[112:])
        expected_rows=32;fixture_name="synthetic_tiny_integration_fixture"
    def run(method):
        command=[sys.executable,str(ROOT/"scripts/run_task.py"),"--method",method,"--input",str(source),
                 "--output",str(directory/method),"--params",json.dumps(PARAMS[method]),"--seed","13"]
        with (directory/(method+".log")).open("w") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        if result.returncode:
            return {"method":method,"status":"failed","exit_code":result.returncode}
        record=json.loads((directory/method/"result.json").read_text())
        prediction=np.load(directory/method/"predictions.npz")["prediction"]
        if prediction.shape != (expected_rows,) or not np.isfinite(prediction).all():
            raise RuntimeError(method+" invalid predictions")
        return {k:record[k] for k in ["method","status","rmse","fit_wall_sec","predict_wall_sec"]}
    with ThreadPoolExecutor(max_workers=4) as workers:
        records=list(workers.map(run,PARAMS))
    report={"scope":"local integration smoke only; not a rebuttal benchmark result", "models":records,
            "fixture":fixture_name,"output_directory":str(directory.relative_to(ROOT)),"crossgb_fit_calls":0}
    (directory/"summary.json").write_text(json.dumps(report,indent=2))
    (ROOT/"reports/latest_smoke.json").write_text(json.dumps(report,indent=2))
    if args.data:
        (ROOT/"reports/latest_real_smoke.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if all(r["status"]=="success" for r in records) else 1


if __name__=="__main__":
    raise SystemExit(main())
