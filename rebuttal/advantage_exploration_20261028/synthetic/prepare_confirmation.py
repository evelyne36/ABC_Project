#!/usr/bin/env python3
"""Generate only preregistered A06 confirmation draws after a complete HPO freeze."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import prepare_synthetic as dev

ALLOWED_CASE = 'A06_two_sparse_modules'
EXPECTED_CASES = {ALLOWED_CASE, 'gas_turbine_nox'}
EXPECTED_METHODS = {'CrossGB','XGB','LGB_constant','LGB_linear','PL','LinXGBoost'}
HERE = Path(__file__).resolve().parent


def params_hash(params):
    payload=json.dumps(params,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_freeze(path, declaration, protocol, now=None):
    """Fail closed on missing cases, methods, selected params, hashes, or time."""
    require(path.is_file(),'Missing root confirmation_freeze.json; confirmation remains sealed.')
    freeze=json.loads(path.read_text())
    require(set(freeze.get('cases',[]))==EXPECTED_CASES,'Freeze must declare exactly both selected cases.')
    methods=freeze.get('methods',[])
    require(len(methods)==len(EXPECTED_METHODS) and set(methods)==EXPECTED_METHODS,
            'Freeze must declare all six comparison methods exactly once.')
    stamp=freeze.get('frozen_utc')
    require(isinstance(stamp,str) and bool(stamp),'Freeze requires frozen_utc.')
    frozen=datetime.fromisoformat(stamp.replace('Z','+00:00'))
    require(frozen.tzinfo is not None,'frozen_utc must include a timezone.')
    created=datetime.fromisoformat(declaration['created_utc'])
    require(created<=frozen<=(now or datetime.now(timezone.utc)),
            'Freeze time must follow DEV preregistration and must not be in the future.')
    choices=freeze.get('choices')
    require(isinstance(choices,dict) and set(choices)==EXPECTED_CASES,
            'Freeze choices must cover both cases.')
    for case in sorted(EXPECTED_CASES):
        selected=choices[case]
        require(isinstance(selected,dict) and set(selected)==EXPECTED_METHODS,
                f'Incomplete method choices for {case}.')
        for method,choice in selected.items():
            require(isinstance(choice,dict),f'Invalid choice for {case}/{method}.')
            params=choice.get('params')
            require(isinstance(params,dict) and bool(params),f'Missing selected params for {case}/{method}.')
            require(choice.get('params_sha256')==params_hash(params),
                    f'Selected parameter hash mismatch for {case}/{method}.')
    return freeze,protocol.digest(path)


def preflight(root):
    protocol,source=dev.load_protocol(root)
    declaration=json.loads((HERE/'preregistration.json').read_text())
    require(declaration['version']==dev.VERSION,'Frozen synthetic source version mismatch.')
    require(declaration['generator_sha256']==protocol.digest(HERE/'prepare_synthetic.py'),
            'DEV generator changed after preregistration.')
    require(declaration['reused_protocol_sha256']==protocol.digest(source),
            'Shared save/split protocol changed after preregistration.')
    require(declaration['reserved_confirmation_seeds']==dev.RESERVED_CONFIRM_SEEDS==list(range(111001,111006)),
            'Reserved confirmation seed list changed.')
    case=next(c for c in dev.CASES if c['case']==ALLOWED_CASE)
    require(case==next(c for c in declaration['cases'] if c['case']==ALLOWED_CASE),
            'A06 definition differs from preregistration.')
    require(dev.GENERATOR_NAMESPACE==declaration['generator_namespace'], 'Data namespace changed.')
    declared_methods=json.loads((HERE.parent/'exploration_protocol.json').read_text())['methods']
    require(len(declared_methods)==6 and set(declared_methods)==EXPECTED_METHODS,
            'Exploration method registry differs from the six frozen comparisons.')
    freeze_path=HERE.parent/'confirmation_freeze.json'
    freeze,freeze_sha=validate_freeze(freeze_path,declaration,protocol)
    return protocol,declaration,case,freeze,freeze_path,freeze_sha


def generate(root, check_only=True):
    protocol,declaration,case,freeze,freeze_path,freeze_sha=preflight(root)
    if check_only:
        return dict(status='freeze_validated_no_data_generated',case=ALLOWED_CASE,
                    seeds=dev.RESERVED_CONFIRM_SEEDS,freeze_sha256=freeze_sha)
    out=HERE/'confirmation'
    require(not out.exists(),'Confirmation output already exists; refusing overwrite or partial resume.')
    # This template is metadata only; no DEV or confirmation labels are read.
    template=json.loads((HERE/'data'/f'{ALLOWED_CASE}__s{dev.DEV_SEEDS[0]}.json').read_text())
    out.mkdir(exist_ok=False)
    entries=[]
    for seed in dev.RESERVED_CONFIRM_SEEDS:
        fullX,fullnoise=dev.base_data(seed)
        X=fullX[:case['n'],:case['d']].copy()
        eps=fullnoise[:case['n']].copy()
        signal,region=dev.signal_for(case,X)
        splits=protocol.random_split(case['n'],seed)
        meta=deepcopy(template)
        meta.update(seed=seed,phase='confirmation',
                    confirmation_freeze_file=str(freeze_path.relative_to(root)),
                    confirmation_freeze_sha256=freeze_sha,
                    frozen_utc=freeze['frozen_utc'],
                    confirmation_generator_sha256=protocol.digest(Path(__file__)))
        meta['matching'].update(X_sha256=protocol.ahash(X),eps_sha256=protocol.ahash(eps))
        name=f'{ALLOWED_CASE}__s{seed}'
        saved=protocol.save(out/'data',name,X,signal+eps,splits,meta,
            extra={'oracle_signal':signal,'oracle_region':region,'noise':eps})
        # Verify written arrays directly without reporting held-out summaries.
        import numpy as np
        with np.load(out/'data'/saved['npz'],allow_pickle=False) as arrays:
            require(np.array_equal(arrays['X'],X.astype(np.float32)),'Written covariates differ.')
            require(np.array_equal(arrays['oracle_signal'],signal),'Written signal differs.')
            require(np.array_equal(arrays['y'],signal+eps),'Written response differs.')
            for split,idx in zip(('train','val','test'),splits):
                require(np.array_equal(arrays[split],idx),'Written split differs.')
        entries.append(dict(case=ALLOWED_CASE,data_npz='data/'+saved['npz'],sidecar='data/'+saved['metadata'],
            seed=seed,phase='confirmation',n=case['n'],d=case['d'],formula=case['formula'],
            npz_sha256=saved['npz_sha256'],pair_group=case['pair_group']))
    # Abort the release manifest if the selected configs were concurrently edited.
    require(protocol.digest(freeze_path)==freeze_sha,'Freeze changed during generation.')
    manifest=dict(version=dev.VERSION,source_version=dev.VERSION,case=ALLOWED_CASE,
        confirmation_seeds=dev.RESERVED_CONFIRM_SEEDS,dataset_count=len(entries),entries=entries,
        confirmation_generated=True,generated_utc=datetime.now(timezone.utc).isoformat(),
        confirmation_freeze_file=str(freeze_path.relative_to(root)),confirmation_freeze_sha256=freeze_sha,
        frozen_utc=freeze['frozen_utc'],generator='prepare_confirmation.py',
        generator_sha256=protocol.digest(Path(__file__)),
        development_generator_sha256=declaration['generator_sha256'],
        reused_protocol_sha256=declaration['reused_protocol_sha256'],
        verification='written X, signal, response and splits checked directly; no held-out summaries',
        no_model_results=True)
    protocol.dump(out/'manifest.json',manifest)
    return dict(status='generated',case=ALLOWED_CASE,dataset_count=len(entries),
                manifest=str((out/'manifest.json').relative_to(root)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE.parents[2])
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true',help='Default: validate the real root freeze; never draw or write data.')
    mode.add_argument('--generate',action='store_true',help='Explicitly release the five reserved A06 datasets after a valid freeze.')
    args=parser.parse_args()
    print(json.dumps(generate(args.root.resolve(),check_only=not args.generate)))
