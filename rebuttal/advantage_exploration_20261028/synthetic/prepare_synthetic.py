#!/usr/bin/env python3
"""Predeclared synthetic DEV data only; no fitting, downloading, or confirmation mode."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
from statistics import NormalDist
import subprocess

import numpy as np

VERSION = 'cgb-advantage-synthetic-v1-20261028'
DEV_SEEDS = [110001, 110002]
RESERVED_CONFIRM_SEEDS = [111001, 111002, 111003, 111004, 111005]
NOISE_SD = 0.3
GENERATOR_NAMESPACE = 2026102801
CASES = [
    dict(case='A01_s1_n120_d32', n=120, d=32, kind='sparse',
         formula='f(X)=2 sign(x0) x1', axis='small sample size',
         pair_group='n120_d32', gate_features=[0], predictive_by_region={'0':[1], '1':[1]}),
    dict(case='A02_s1_n240_d8', n=240, d=8, kind='sparse',
         formula='f(X)=2 sign(x0) x1', axis='few irrelevant coordinates',
         pair_group='n240_d8', gate_features=[0], predictive_by_region={'0':[1], '1':[1]}),
    dict(case='A03_s1_n240_d64', n=240, d=64, kind='sparse',
         formula='f(X)=2 sign(x0) x1', axis='many irrelevant coordinates',
         pair_group='n240_d64', gate_features=[0], predictive_by_region={'0':[1], '1':[1]}),
    dict(case='A04_imbalanced_gate', n=240, d=32, kind='imbalanced',
         formula='f(X)=2 [(I(x0>=Phi^-1(.8))-.2)/sqrt(.2*.8)] x1',
         axis='20/80 region imbalance with population-centered gate', pair_group='n240_d32',
         gate_features=[0], predictive_by_region={'0':[1], '1':[1]}),
    dict(case='A05_predictor_switch', n=240, d=32, kind='switch',
         formula='f(X)=-2 x1 if x0<0, else +2 x2',
         axis='one-level gate switches the independent predictive coordinate', pair_group='n240_d32',
         gate_features=[0], predictive_by_region={'0':[1], '1':[2]}),
    dict(case='A06_two_sparse_modules', n=240, d=32, kind='modules',
         formula='f(X)=sqrt(2) [sign(x0) x2 + sign(x1) x3]',
         axis='two additive gate modules, one predictive variable per module', pair_group='n240_d32',
         gate_features=[0,1], predictive_modules=[[2],[3]]),
    dict(case='A07_dense_control', n=120, d=32, kind='dense',
         formula='f(X)=2 sign(x0) sum(x1,...,x31)/sqrt(31)',
         axis='matched dense-slope control for A01', pair_group='n120_d32',
         gate_features=[0], predictive_by_region={'0':list(range(1,32)), '1':list(range(1,32))}),
    dict(case='A08_intercept_control', n=120, d=32, kind='intercept',
         formula='f(X)=2 sign(x0)', axis='matched region-intercept control for A01',
         pair_group='n120_d32', gate_features=[0], predictive_by_region={'0':[], '1':[]}),
]


def load_protocol(root):
    source = root / 'rebuttal/fastlane_20261027/prepare_data.py'
    spec = importlib.util.spec_from_file_location('fastlane_data_protocol', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION = VERSION
    return module, source


def audit_historical_seeds(root):
    """Conservative integer-token scan; decimal fragments are not seed use.

    The entire new exploration directory is excluded because sibling agents may
    concurrently register this same predeclared seed set there.
    """
    pattern = r'(?<![0-9.])(?:' + '|'.join(map(str, DEV_SEEDS+RESERVED_CONFIRM_SEEDS)) + r')(?![0-9.])'
    command = ['rg', '-l', '--pcre2', pattern, 'rebuttal', 'local_sync']
    for ext in ('json','jsonl','py','md','csv','yaml','yml','sh','toml','txt'):
        command += ['--glob', '*.'+ext]
    command += ['--glob', '!**/advantage_exploration_20261028/**']
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if result.returncode not in (0,1):
        raise RuntimeError(result.stderr)
    collisions = result.stdout.splitlines()
    if collisions:
        raise RuntimeError('Historical integer seed-token collision: '+json.dumps(collisions))
    # Also inspect filenames of existing generated datasets and seed directories.
    filename_pattern = re.compile(r'(?<![0-9])(?:s|seed[_=-]?)?(' + '|'.join(map(str,DEV_SEEDS+RESERVED_CONFIRM_SEEDS)) + r')(?![0-9])')
    existing_files = subprocess.run(['rg','--files','--hidden','rebuttal','local_sync',
        '--glob','!**/advantage_exploration_20261028/**'], cwd=root, text=True,
        capture_output=True, check=True).stdout.splitlines()
    filename_collisions = [p for p in existing_files if filename_pattern.search(p)]
    if filename_collisions:
        raise RuntimeError('Historical seed filename collision: '+json.dumps(filename_collisions))
    return dict(seeds=DEV_SEEDS+RESERVED_CONFIRM_SEEDS, collisions=[],
                searched_roots=['rebuttal','local_sync'],
                excluded='rebuttal/advantage_exploration_20261028 (current new exploration)',
                text_scope='json/jsonl/py/md/csv/yaml/yml/sh/toml/txt; integer tokens, excluding decimal substrings',
                filenames_checked=len(existing_files), filename_collisions=[])


def base_data(seed):
    # A common largest array gives nested covariates and common noise across
    # dimension/sample-size axes without dependence on case iteration order.
    X = np.random.default_rng(np.random.SeedSequence([seed,GENERATOR_NAMESPACE,1])).normal(size=(480,64))
    eps = np.random.default_rng(np.random.SeedSequence([seed,GENERATOR_NAMESPACE,2])).normal(scale=NOISE_SD,size=480)
    return X,eps


def signal_for(case, X):
    sign = np.where(X[:,0] >= 0, 1., -1.)
    region = (X[:,0] >= 0).astype(np.int64)
    kind = case['kind']
    if kind == 'sparse':
        signal = 2*sign*X[:,1]
    elif kind == 'imbalanced':
        region = (X[:,0] >= NormalDist().inv_cdf(.8)).astype(np.int64)
        signal = 2*(region-.2)/np.sqrt(.2*.8)*X[:,1]
    elif kind == 'switch':
        signal = 2*sign*X[np.arange(len(X)),1+region]
    elif kind == 'modules':
        sign2 = np.where(X[:,1] >= 0,1.,-1.)
        region = 2*region+(X[:,1]>=0).astype(np.int64)
        signal = np.sqrt(2)*(sign*X[:,2]+sign2*X[:,3])
    elif kind == 'dense':
        signal = 2*sign*X[:,1:].sum(axis=1)/np.sqrt(X.shape[1]-1)
    elif kind == 'intercept':
        signal = 2*sign
    else:
        raise ValueError(kind)
    return signal,region


def generate(root, out):
    protocol,source = load_protocol(root)
    audit = audit_historical_seeds(root)
    if list((out/'data').glob('*.npz')):
        raise RuntimeError('DEV data already exists; use verification rather than overwrite.')
    preregistration = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        status='frozen before generation; no model scores read', cases=CASES,
        development_seeds=DEV_SEEDS, reserved_confirmation_seeds=RESERVED_CONFIRM_SEEDS,
        confirmation_status='registered only; not generated; generate/evaluate only after HPO freeze',
        generator_namespace=GENERATOR_NAMESPACE, noise_sd=NOISE_SD,
        population_signal_L2=2.0, population_signal_second_moment=4.0,
        seed_audit=audit, generator_sha256=protocol.digest(Path(__file__)),
        reused_protocol_source=str(source.relative_to(root)), reused_protocol_sha256=protocol.digest(source))
    protocol.dump(out/'preregistration.json',preregistration)
    entries=[]
    for seed in DEV_SEEDS:
        fullX,fullnoise=base_data(seed)
        for case in CASES:
            X=fullX[:case['n'],:case['d']].copy()
            eps=fullnoise[:case['n']].copy()
            signal,region=signal_for(case,X)
            splits=protocol.random_split(len(X),seed)
            name=f"{case['case']}__s{seed}"
            meta=dict(seed=seed, phase='development', task_type='regression', family=case['case'],
                case=case['case'], control=case['kind'], source='independent standard Normal coordinates; fresh versioned data seeds',
                source_version=VERSION, formula=case['formula']+'; y=f(X)+epsilon',
                case_definition=case, noise_sd=NOISE_SD, feature_names=[f'x{i}' for i in range(case['d'])],
                population_signal_L2=2.0, population_signal_mean=0.0, population_signal_variance=4.0,
                preprocessing={'X':'none; all coordinates population N(0,1)', 'y':'none; analytic population scaling only'},
                matching={'pair_group':case['pair_group'], 'X_sha256':protocol.ahash(X), 'eps_sha256':protocol.ahash(eps),
                          'shared_latent_base':'same seed shares 480x64 master X and independent 480-vector noise; subsets by n,d'},
                split_protocol='seeded row random 60/20/20; source fastlane random_split unchanged',
                oracle_support={k:v for k,v in case.items() if k in ('gate_features','predictive_by_region','predictive_modules')},
                oracle_arrays='diagnostic only; never concatenate oracle_signal/oracle_region/noise to X',
                gate_definition={'imbalanced_probability':.2, 'threshold':NormalDist().inv_cdf(.8),
                                 'threshold_scope':'fixed analytic population quantile; no sample labels or target use'} if case['kind']=='imbalanced' else {'threshold':0.0},
                seed_collision_check=audit)
            saved=protocol.save(out/'data',name,X,signal+eps,splits,meta,
                                extra={'oracle_signal':signal,'oracle_region':region,'noise':eps})
            entries.append(dict(case=case['case'], data_npz='data/'+saved['npz'],
                sidecar='data/'+saved['metadata'], seed=seed, phase='development',
                n=case['n'],d=case['d'],formula=case['formula'],npz_sha256=saved['npz_sha256'],
                pair_group=case['pair_group']))
    manifest=dict(version=VERSION, source_version=VERSION, preregistration='preregistration.json',
        generator='prepare_synthetic.py', data_protocol='data_protocol.md',
        development_seeds=DEV_SEEDS, confirmation_seeds_reserved=RESERVED_CONFIRM_SEEDS,
        confirmation_generated=False, confirmation_release_condition='HPO and selection rule frozen before generating/evaluating confirmation data',
        case_count=len(CASES), dataset_count=len(entries), entries=entries,
        paired_controls={'A01_s1_n120_d32':['A07_dense_control','A08_intercept_control']},
        no_model_results=True)
    protocol.dump(out/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    manifest=generate(args.root.resolve(),args.out.resolve())
    print(json.dumps({'version':VERSION,'dev_npz':manifest['dataset_count'],'confirmation_generated':False}))


if __name__=='__main__':
    main()
