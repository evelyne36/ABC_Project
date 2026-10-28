from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_nested_runner_is_plan_only_without_execute_and_counts_true_fits(tmp_path):
    params=tmp_path/"candidates.json"
    params.write_text('[{"n_estimators":2,"max_depth":1,"lbda":0.1}]')
    out=tmp_path/"nested"
    command=[sys.executable,str(ROOT/"scripts/run_nested.py"),"--method","linxgboost",
             "--data",str(ROOT/"inputs/moneyball"),"--candidate-file",str(params),
             "--outer-fold","1","--output",str(out)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    plan=json.loads(result.stdout)
    assert plan["planned_fits"]==5 and plan["crossgb_fit_calls"]==0
    assert not out.exists()
    execution=subprocess.run(command+["--execute","--workers","2","--label","integration_check"],capture_output=True,text=True,timeout=120)
    assert execution.returncode==0,execution.stdout+execution.stderr
    record=json.loads((out/"outer1/summary.json").read_text())
    assert record["status"]=="success" and record["selected_candidate_index"]==1
    assert record["inner_folds"]==4 and record["candidate_count"]==1
    assert record["budget_matches_archived_cgb_candidate_count"] is False
    assert record["label"]=="integration_check"
    resumed=subprocess.run(command+["--execute","--resume","--workers","2","--label","integration_check"],capture_output=True,text=True,timeout=60)
    assert resumed.returncode==0,resumed.stdout+resumed.stderr
    record=json.loads((out/"outer1/summary.json").read_text())
    assert record["new_fit_processes"]==0
    # A cached input must not drift away from the original frozen partition.
    import numpy as np
    input_file=out/"outer1/inner1.npz"
    with np.load(input_file,allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    arrays["y_train"][0]+=100
    np.savez_compressed(input_file,**arrays)
    changed=subprocess.run(command+["--execute","--resume","--label","integration_check"],capture_output=True,text=True,timeout=60)
    assert changed.returncode!=0
    assert "differs from frozen protocol" in changed.stderr
