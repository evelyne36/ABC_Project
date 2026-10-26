#!/usr/bin/env python3
"""把远端 dump（R <fam> <relpath> || k=v; k=v）解析成宽表 CSV。

用法：
  python3 aggregate2.py /tmp/n2collect/raw_*.txt
输出（本目录）：
  dump_long.csv   长表：fam, relpath, key, value
  dump_metric.csv 只留指标类键（含 rmse/mae/corr/r2）的宽表
"""
import csv
import glob
import os
import sys
from collections import defaultdict

METRIC_HINTS = ("rmse", "mae", "r2", "corr", "auc", "loss", "score")


def parse_line(line):
    body = line[2:]
    if "||" not in body:
        return None
    path, kv = body.split("||", 1)
    parts = path.split()
    if len(parts) < 2:
        return None
    fam, rel = parts[0], parts[1]
    d = {}
    for seg in kv.split(";"):
        if "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        d[k.strip()] = v.strip()
    return fam, rel, d


def main(argv):
    outdir = os.path.dirname(os.path.abspath(__file__))
    files = []
    for pat in argv[1:] or ["/tmp/n2collect/raw_*.txt"]:
        files.extend(glob.glob(pat))
    if not files:
        sys.exit("no raw files matched")

    long_rows = []
    metric_rows = []
    for f in files:
        host = os.path.basename(f).replace("raw_", "").replace(".txt", "")
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("R "):
                    continue
                got = parse_line(line.rstrip("\n"))
                if not got:
                    continue
                fam, rel, d = got
                keys = [k for k in d if any(h in k.lower() for h in METRIC_HINTS)]
                row = {"fam": fam, "relpath": rel, "host": host}
                for k, v in d.items():
                    long_rows.append({"fam": fam, "relpath": rel, "host": host, "key": k, "value": v})
                for k in keys:
                    row[k] = d[k]
                if keys:
                    metric_rows.append(row)

    def dump(name, rows):
        if not rows:
            return 0
        cols = sorted({k for r in rows for k in r})
        with open(os.path.join(outdir, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return len(rows)

    n1 = dump("dump_long.csv", long_rows)
    n2 = dump("dump_metric.csv", metric_rows)
    fams = defaultdict(int)
    for r in metric_rows:
        fams[r["fam"]] += 1
    print(f"long={n1} metric={n2} fams={dict(fams)}")


if __name__ == "__main__":
    main(sys.argv)
