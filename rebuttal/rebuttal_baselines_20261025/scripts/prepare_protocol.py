#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.protocol import REAL,SYNTHETIC,prepare_dataset


def main():
    parser=argparse.ArgumentParser(description="Prepare historical five-outer/four-inner regression inputs; zero model fits")
    parser.add_argument("--dataset",choices=REAL+SYNTHETIC+["all"],required=True)
    parser.add_argument("--project-root",type=Path,default=ROOT.parent)
    parser.add_argument("--output-root",type=Path,default=ROOT/"inputs")
    args=parser.parse_args()
    datasets=REAL+SYNTHETIC if args.dataset=="all" else [args.dataset]
    for name in datasets:
        record=prepare_dataset(name,args.output_root/name,args.project_root)
        print(json.dumps({k:record[k] for k in ["dataset","rows","features","archived_split_entries_matched"]}))


if __name__=="__main__":
    main()
