"""Materialize historical inputs and folds, without importing/fitting CrossGB."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import KFold

REAL=["kin8nm","moneyball","cpu_activity","houses","energy_efficiency"]
SYNTHETIC=["regime_slope","feature_switch","sparse_multiplicative","curved_control"]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_hash(index):
    return hashlib.sha256(np.asarray(index,dtype=np.int64).tobytes()).hexdigest()


def prepare_dataset(dataset, output, project):
    output,project=Path(output),Path(project)
    if dataset not in REAL+SYNTHETIC:
        raise ValueError("Unknown planned regression dataset: "+dataset)
    full=project/"aaai_FullCodeProject-check_codereview0701"
    sys.path.insert(0,str(full))
    try:
        if dataset in REAL:
            registry=json.loads((project/"rebuttal_server_config_20261025/inputs/dataset_registry.json").read_text())
            entry=next(r for r in registry["tabular"] if r["dataset"]==dataset)
            path=project/entry["project_relative_path"]
            if file_hash(path)!=entry["sha256"]:
                raise RuntimeError("Archived input snapshot hash mismatch: "+dataset)
            module_name="random600.utils.runner" if dataset=="houses" else "tabular_extra_random300.utils.runner"
            upstream=importlib.import_module(module_name)
            data=upstream.load_dataset(path,"regression","onehot")
            X,y=data.X,data.y
            args=argparse.Namespace(outer_folds=5,inner_folds=4,seed=42)
            outer=upstream.split_indices(data,args)
            inner={k:upstream.inner_split_indices(data,X[t],y[t],args,k) for k,(t,_) in enumerate(outer)}
            provenance={"source_csv":entry["project_relative_path"],"source_csv_sha256":entry["sha256"],
                        "loader":module_name,"loader_sha256":file_hash(Path(upstream.__file__)),
                        "outer_seed":42,"inner_seed_rule":"42 + zero-based outer index", "categorical_handling":"onehot",
                        "encoding_scope":"historical full-input onehot schema; do not describe as fold-local encoder fitting",
                        "raw_feature_columns":entry["raw_feature_columns"],
                        "historical_cgb_candidates":600 if dataset=="houses" else 300,
                        "split_match_scope":"original source functions and recorded protocol seed; no archived real index manifest recovered"}
            expected=None
        else:
            upstream=importlib.import_module("synthetic_metagb.scenarios")
            X,y=upstream.generate_instance(dataset,42,3000,0.15,0,10)
            outer=list(KFold(5,shuffle=True,random_state=314159).split(X))
            inner={k:list(KFold(4,shuffle=True,random_state=314159+k+1).split(X[t])) for k,(t,_) in enumerate(outer)}
            source=full/"datacollection0702/synthetic_main1020full_random300/split_manifest.json"
            expected={r["outer_fold"]:r for r in json.loads(source.read_text())["entries"] if r["scenario"]==dataset+"__dim10" and r["data_seed"]==42}
            provenance={"generator":"synthetic_metagb.scenarios.generate_instance", "generator_sha256":file_hash(Path(upstream.__file__)),
                        "data_seed":42,"n_samples":3000,"noise":0.15,"target_features":10,
                        "outer_seed":314159,"inner_seed_rule":"314159 + one-based outer index", "historical_cgb_candidates":300,
                        "archived_split_manifest":str(source.relative_to(project)),"archived_split_manifest_sha256":file_hash(source),
                        "split_match_scope":"outer and inner indices exactly hash-matched to archived main experiment; generated features use recorded source snapshot"}
    finally:
        sys.path.remove(str(full))
    arrays={"X":X,"y":y};rows=[];matched=0
    for k,(train,test) in enumerate(outer,1):
        row={"outer_fold":k,"train":f"outer{k}_train","test":f"outer{k}_test", "train_hash":index_hash(train),"test_hash":index_hash(test),"inner":[]}
        arrays[row["train"]]=train;arrays[row["test"]]=test
        for j,(it,iv) in enumerate(inner[k-1],1):
            at,av=train[it],train[iv]
            item={"inner_fold":j,"train":f"outer{k}_inner{j}_train","validation":f"outer{k}_inner{j}_validation", "train_hash":index_hash(at),"validation_hash":index_hash(av)}
            arrays[item["train"]]=at;arrays[item["validation"]]=av;row["inner"].append(item)
        if expected is not None:
            old=expected[k]
            if row["train_hash"]!=old["train_hash"] or row["test_hash"]!=old["test_hash"]:
                raise RuntimeError("Synthetic outer split mismatch")
            if any(item["train_hash"]!=ref["train_hash"] or item["validation_hash"]!=ref["validation_hash"] for item,ref in zip(row["inner"],old["inner"])):
                raise RuntimeError("Synthetic inner split mismatch")
            matched+=1
        rows.append(row)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/"data.npz",**arrays)
    manifest={"dataset":dataset,"task":"regression","rows":len(X),"features":X.shape[1],"dtype":str(X.dtype),
              "model_fit_calls":0,"source":provenance,"outer_folds":rows,"data_npz_sha256":file_hash(output/"data.npz"),
              "archived_split_entries_matched":matched}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def load_protocol(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    if file_hash(directory/"data.npz")!=manifest["data_npz_sha256"]:
        raise RuntimeError("Prepared data checksum mismatch")
    with np.load(directory/"data.npz",allow_pickle=False) as data:
        X,y=data["X"].copy(),data["y"].copy();folds=[]
        for row in manifest["outer_folds"]:
            fold={"outer_fold":row["outer_fold"],"train":data[row["train"]].copy(),"test":data[row["test"]].copy(),"inner":[]}
            if index_hash(fold["train"])!=row["train_hash"] or index_hash(fold["test"])!=row["test_hash"]:
                raise RuntimeError("Prepared outer index checksum mismatch")
            for inner in row["inner"]:
                t,v=data[inner["train"]].copy(),data[inner["validation"]].copy()
                if index_hash(t)!=inner["train_hash"] or index_hash(v)!=inner["validation_hash"]:
                    raise RuntimeError("Prepared inner index checksum mismatch")
                fold["inner"].append({"train":t,"validation":v})
            folds.append(fold)
    return X,y,folds,manifest
