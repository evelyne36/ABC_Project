#!/usr/bin/env python3
"""N1 逐折聚合：读 raw_<host>.txt -> 去重 -> 均值/中位数/失败候选数。只读，不写训练产物。"""
import glob, collections, csv, statistics, os
rows = {}
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'raw_*.txt'))):
    host = os.path.basename(f)[4:-4]
    for ln in open(f):
        if not ln.startswith('R '):
            continue
        p = ln.split()
        m, d, fold, st = p[1], p[2], p[3], p[4]
        r = dict(method=m, dataset=d.replace('_compat',''), fold=fold, status=st,
                 test_rmse=float(p[5]), failed_candidates=int(p[6]), host=host,
                 variant=('compat' if d.endswith('_compat') else 'orig'))
        k = (m, r['dataset'], fold)
        # 去重：TAO 优先 _compat；其余先到先得
        if k not in rows or r['variant'] == 'compat':
            rows[k] = r
out = sorted(rows.values(), key=lambda r: (r['method'], r['dataset'], r['fold']))
with open(os.path.join(os.path.dirname(__file__), 'N1_folds.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['method','dataset','fold','status','test_rmse','failed_candidates','host','variant'])
    w.writeheader(); w.writerows(out)
print(f"{'method':26s} {'dataset':16s} {'n':>2s} {'mean':>10s} {'median':>10s} {'maxfail':>7s}")
g = collections.defaultdict(list)
for r in out: g[(r['method'], r['dataset'])].append(r)
for k in sorted(g):
    v = g[k]; rs = [x['test_rmse'] for x in v]
    print(f"{k[0]:26s} {k[1]:16s} {len(v):2d} {sum(rs)/len(rs):10.6f} {statistics.median(rs):10.6f} {max(x['failed_candidates'] for x in v):7d}")
