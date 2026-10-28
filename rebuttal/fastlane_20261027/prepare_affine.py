#!/usr/bin/env python3
"""Add S1 affine regional-intercept controls without mutating existing data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS=(97001,97002,98001,98002,98003,98004,98005)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ahash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()

def prepare(data):
    destinations=[data/f'S1_affine__s{s}{ext}' for s in SEEDS for ext in ('.npz','.json')]
    existing=[str(p) for p in destinations if p.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite: '+', '.join(existing))
    sources=[data/f'S1_sparse__s{s}{ext}' for s in SEEDS for ext in ('.npz','.json')]
    before={p:sha(p) for p in sources}
    created=[]
    for seed in SEEDS:
        source=data/f'S1_sparse__s{seed}.npz'
        with np.load(source,allow_pickle=False) as loaded:
            arrays={key:loaded[key].copy() for key in loaded.files}
        meta=json.loads(source.with_suffix('.json').read_text())
        sign=np.where(arrays['X'][:,0]>=0,1.,-1.)
        signal=(arrays['oracle_signal']+2*sign)/np.sqrt(2.)
        arrays['oracle_signal']=signal
        arrays['y']=signal+arrays['noise']
        for split in ('train','val','test'):
            arrays['y_'+split]=arrays['y'][arrays[split]]
        assert np.isfinite(arrays['y']).all()
        name=f'S1_affine__s{seed}'
        path=data/(name+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        meta.update(dataset=name,control='affine_offset',
                    protocol_version='fastlane-v2-20261027-S1-affine-addition',
                    formula='(2 sign(x0) x1 + 2 sign(x0))/sqrt(2) + eps',
                    derivation='oracle_signal_new = (S1_sparse.oracle_signal + 2 sign(S1_sparse.X[:,0]))/sqrt(2); y_new = oracle_signal_new + S1_sparse.noise',
                    population_signal_variance=4.,
                    oracle_support={'gate_features':[0],'predictive_by_region':{'0':[1],'1':[1]},
                                    'slopes':[-float(np.sqrt(2)),float(np.sqrt(2))],
                                    'intercepts':[-float(np.sqrt(2)),float(np.sqrt(2))]},
                    npz_sha256=sha(path),array_sha256={k:ahash(v) for k,v in arrays.items()},
                    source_sparse_npz=source.name,source_sparse_npz_sha256=before[source],
                    note='Slope plus regional intercept control for N5. Existing S1_offset remains an immutable pure-step control (2 sign(x0)+eps).')
        meta['matching'].update(population_signal_variance=4.,shared_X_noise_splits_with=source.name,
                                stored_X_sha256=ahash(arrays['X']))
        with path.with_suffix('.json').open('x') as stream:
            stream.write(json.dumps(meta,indent=2,allow_nan=False)+'\n')
        created.append({'dataset':name,'npz':str(path),'npz_sha256':meta['npz_sha256'],
                        'metadata':str(path.with_suffix('.json'))})
    assert all(sha(p)==digest for p,digest in before.items()),'Existing sparse inputs changed'
    return created

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[2]/'local_sync/fastlane_data')
    args=p.parse_args()
    print(json.dumps({'created':prepare(args.data.resolve()),'existing_files_modified':0},indent=2))

if __name__=='__main__':main()
