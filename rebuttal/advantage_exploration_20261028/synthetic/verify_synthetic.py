#!/usr/bin/env python3
"""Direct, outcome-free artifact checks; no estimators or held-out statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import prepare_synthetic as gen


def verify(root, folder):
    protocol,_ = gen.load_protocol(root)
    manifest=json.loads((folder/'manifest.json').read_text())
    declaration=json.loads((folder/'preregistration.json').read_text())
    assert manifest['version']==declaration['version']==gen.VERSION
    assert declaration['generator_sha256']==protocol.digest(folder/'prepare_synthetic.py')
    assert declaration['reused_protocol_sha256']==protocol.digest(root/declaration['reused_protocol_source'])
    assert declaration['cases']==gen.CASES
    assert manifest['confirmation_generated'] is False
    assert manifest['confirmation_seeds_reserved']==gen.RESERVED_CONFIRM_SEEDS
    entries=manifest['entries']
    assert len(entries)==16 and manifest['case_count']==8
    assert sorted(p.name for p in (folder/'data').glob('*.npz'))==sorted(Path(e['data_npz']).name for e in entries)
    assert len(list((folder/'data').glob('*.json')))==16
    loaded={}
    checks=[]
    for entry in entries:
        assert entry['phase']=='development' and entry['seed'] in gen.DEV_SEEDS
        case=next(c for c in gen.CASES if c['case']==entry['case'])
        path=folder/entry['data_npz']
        meta=json.loads((folder/entry['sidecar']).read_text())
        assert protocol.digest(path)==entry['npz_sha256']==meta['npz_sha256']
        with np.load(path,allow_pickle=False) as data:
            arrays={k:data[k] for k in data.files}
        assert {k:protocol.ahash(v) for k,v in arrays.items()}==meta['array_sha256']
        assert arrays['X'].shape==(case['n'],case['d'])
        assert arrays['X'].dtype==np.float32 and arrays['y'].dtype==np.float64
        assert meta['features']==case['d'] and meta['oracle_support_in_X'] is False
        assert meta['population_signal_variance']==4.0 and meta['noise_sd']==.3
        assert all(np.isfinite(v).all() for k,v in arrays.items() if v.dtype.kind in 'fiu')
        X,eps=gen.base_data(entry['seed'])
        X=X[:case['n'],:case['d']]; eps=eps[:case['n']]
        expected_signal,expected_region=gen.signal_for(case,X)
        assert np.array_equal(arrays['X'],X.astype(np.float32))
        assert np.array_equal(arrays['noise'],eps)
        assert np.array_equal(arrays['oracle_signal'],expected_signal)
        assert np.array_equal(arrays['oracle_region'],expected_region)
        assert np.array_equal(arrays['y'],expected_signal+eps)
        expected_splits=protocol.random_split(case['n'],entry['seed'])
        for name,indices in zip(('train','val','test'),expected_splits):
            assert np.array_equal(arrays[name],indices)
            assert np.array_equal(arrays['y_'+name],arrays['y'][indices])
        joined=np.concatenate([arrays[k] for k in ('train','val','test')])
        assert np.array_equal(np.sort(joined),np.arange(case['n']))
        assert [len(arrays[k]) for k in ('train','val','test')]==[int(.6*case['n']),int(.2*case['n']),int(.2*case['n'])]
        loaded[(case['case'],entry['seed'])]=arrays
        checks.append(dict(case=case['case'],seed=entry['seed'],shape=list(arrays['X'].shape),
                           split_sizes=meta['split_sizes'],hashes_verified=True,formula_verified=True))
    for seed in gen.DEV_SEEDS:
        a=loaded[('A01_s1_n120_d32',seed)]
        for control in ('A07_dense_control','A08_intercept_control'):
            b=loaded[(control,seed)]
            for key in ('X','noise','train','val','test'):
                assert np.array_equal(a[key],b[key]),(control,key)
        low=loaded[('A02_s1_n240_d8',seed)]
        high=loaded[('A03_s1_n240_d64',seed)]
        assert np.array_equal(low['X'],high['X'][:,:8])
        for key in ('noise','oracle_signal','train','val','test'):
            assert np.array_equal(low[key],high[key])
    assert not any(f'__s{seed}' in p.name for p in (folder/'data').iterdir() for seed in gen.RESERVED_CONFIRM_SEEDS)
    report=dict(status='passed',version=gen.VERSION,datasets_checked=len(checks),
                checked=checks,paired_controls_verified=True,nested_dimension_axis_verified=True,
                reserved_confirmation_absent=True,model_fits=0,held_out_label_summaries_emitted=False)
    protocol.dump(folder/'validation.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--folder',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    result=verify(args.root.resolve(),args.folder.resolve())
    print(json.dumps({k:result[k] for k in ('status','datasets_checked','reserved_confirmation_absent','model_fits')}))
