#!/usr/bin/env python3
"""Freeze method-specific search configurations; performs no model fitting."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",choices=["linxgboost","pilot","gbdt_pl","tao"],required=True)
    parser.add_argument("--count",type=int,required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.count<1:
        parser.error("count must be positive")
    rng=np.random.default_rng(args.seed)
    def pick(values):return values[int(rng.integers(len(values)))]
    def log(a,b):return float(np.exp(rng.uniform(np.log(a),np.log(b))))
    # Include an author starting configuration, and cover very small ensembles:
    # linear-leaf boosters can converge with far fewer trees than constant GBDT.
    author_start={
        "linxgboost":{"n_estimators":5,"learning_rate":0.3,"max_depth":6,"lbda":0.0,"min_samples_leaf":2},
        "pilot":{"max_depth":12,"max_model_depth":100,"min_sample_leaf":5,"min_sample_split":10,"rel_tolerance":0.0},
        "gbdt_pl":{"num_trees":500,"learning_rate":0.1,"num_leaves":255,"max_var":5,"l2_reg":0.01,"l1_reg":0.01,"min_sum_hessians":100,"num_bins":63,"leaf_type":"half_additive"},
        "tao":{"depth":3,"max_iter":20,"C_dnode":10.0,"alpha":0.0},
    }
    rows=[author_start[args.method]]
    while len(rows)<args.count:
        if args.method=="linxgboost":
            row={"n_estimators":pick([1,3,5,10,20,50,100,200,400]),"learning_rate":log(0.01,0.5),"max_depth":pick([1,2,3,4,6,8]),"lbda":0.0 if rng.random()<0.1 else log(1e-4,100),"min_samples_leaf":pick([2,5,10,20,50])}
        elif args.method=="pilot":
            row={"max_depth":pick([1,2,4,6,8,12]),"max_model_depth":pick([5,8,20,50,100]),"min_sample_leaf":pick([2,5,10,20,50]),"min_sample_split":pick([5,10,20,40,100]),"rel_tolerance":0.0 if rng.random()<0.1 else log(1e-5,0.05)}
        elif args.method=="gbdt_pl":
            row={"num_trees":pick([10,20,50,100,200,400,800]),"learning_rate":log(0.005,0.3),"num_leaves":pick([4,8,16,31,63,127,255]),"max_var":pick([1,2,3,5]),"l2_reg":log(1e-4,100),"l1_reg":0.0 if rng.random()<0.5 else log(1e-4,1),"min_sum_hessians":pick([2,5,10,20,50,100]),"num_bins":pick([63,127,255]),"leaf_type":"half_additive"}
        else:
            row={"depth":pick([1,2,3,4,5,6,7,8]),"max_iter":pick([10,20,30]),"C_dnode":log(0.01,1000),"alpha":pick([0.0,0.5,1.0])}
        if row not in rows:rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as output:json.dump(rows,output,indent=2)
    print(json.dumps({"method":args.method,"count":len(rows),"search_seed":args.seed,"model_fits":0,"space_scope":"proposed follow-up baseline space; not an author-paper exact search reproduction"}))


if __name__=="__main__":
    main()
