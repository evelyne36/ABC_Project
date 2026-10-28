#!/usr/bin/env python3
"""Check the complete root freeze; --generate materializes five NOX packages.

No model is fitted or scored. DEV generator, manifest and data remain untouched.
"""
from pathlib import Path
import argparse, copy, datetime, hashlib, importlib.util, json, math
import numpy as np

HERE = Path(__file__).resolve().parent
FREEZE = HERE.parent/'confirmation_freeze.json'
CASE = 'gas_turbine_nox'
CASES = {'A06_two_sparse_modules', CASE}
METHODS = {'CrossGB','XGB','LGB_constant','LGB_linear','PL','LinXGBoost'}
SEEDS = list(range(113001,113006))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_hash(params):
    return hashlib.sha256(json.dumps(params,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_freeze(doc):
    """Reject partial choices even when the real case itself is complete."""
    if set(doc.get('cases',[])) != CASES or len(doc['cases']) != 2:
        raise ValueError('Freeze must name exactly both selected cases')
    if set(doc.get('methods',[])) != METHODS or len(doc['methods']) != 6:
        raise ValueError('Freeze must name exactly all six selected methods')
    timestamp=doc.get('frozen_utc')
    if not isinstance(timestamp,str):raise ValueError('Missing frozen_utc timestamp')
    parsed=datetime.datetime.fromisoformat(timestamp.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise ValueError('frozen_utc must include timezone')
    if parsed.utcoffset()!=datetime.timedelta(0):raise ValueError('frozen_utc must be UTC')
    choices=doc.get('choices',{})
    if set(choices)!=CASES:raise ValueError('Freeze choices must cover both cases exactly')
    for case in CASES:
        if not isinstance(choices[case],dict) or set(choices[case])!=METHODS:
            raise ValueError('Incomplete six-method choices for '+case)
        for method,choice in choices[case].items():
            if not isinstance(choice,dict):raise ValueError('Choice must be a mapping')
            params=choice.get('params')
            if not isinstance(params,dict) or not params:raise ValueError('Missing nonempty params for '+case+'/'+method)
            if choice.get('params_sha256')!=canonical_hash(params):raise ValueError('Parameter hash mismatch for '+case+'/'+method)
            candidate=choice.get('candidate')
            if type(candidate) is not int or candidate<0:raise ValueError('Invalid frozen candidate index')
            metric=choice.get('mean_development_primary')
            if isinstance(metric,bool) or not isinstance(metric,(float,int)) or not math.isfinite(metric):
                raise ValueError('Missing finite development selection metric')
    return doc

def import_dev():
    spec=importlib.util.spec_from_file_location('frozen_real_dev',HERE/'prepare_real.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def check_inputs():
    # Freeze is checked before opening source target data.
    doc=validate_freeze(json.loads(FREEZE.read_text()))
    dev_manifest=json.loads((HERE/'manifest.json').read_text())
    if sha(HERE/'prepare_real.py')!=dev_manifest['generator_sha256']:
        raise ValueError('Frozen DEV generator changed')
    dev=import_dev();prior_path=dev.ROOT/dev_manifest['prior_generator_path']
    if sha(prior_path)!=dev_manifest['prior_generator_sha256']:
        raise ValueError('Prior split/preprocessing implementation changed')
    refs=[]
    for e in dev_manifest['entries']:
        if e['case']!=CASE:continue
        if sha(HERE/e['npz'])!=e['npz_sha256']:raise ValueError('Frozen DEV package changed')
        refs.append((e,json.loads((HERE/e['metadata']).read_text())))
    if len(refs)!=2:raise ValueError('Expected the two frozen NOX DEV packages')
    reference=refs[0][1]
    if reference['source']['target']!='NOX' or reference['training_cap']!=600 or reference['features']!=9:
        raise ValueError('Frozen task/target/cap/feature mismatch')
    if sha(HERE/'sources/uci_551.zip')!=reference['source']['archive_sha256']:
        raise ValueError('Official source archive changed')
    if dev_manifest['reserved_confirmation_seeds']!=SEEDS:
        raise ValueError('Reserved confirmation seed list changed')
    return doc,dev,dev_manifest,refs

def generate(doc,dev,dev_manifest,refs):
    output=HERE/'confirmation'
    pending=HERE/'.confirmation_pending'
    if output.exists() or pending.exists():
        raise FileExistsError('Confirmation output or pending generation already exists; no overwrite permitted')
    # Check source/target/split against the immutable DEV references before output.
    prior=dev.prior_runtime()
    X,y,groups,full,source,protocol=dev.load_task(CASE,prior)
    reference=refs[0][1]
    if source!=reference['source'] or list(X.columns)!=reference['feature_names']:
        raise ValueError('Source, target or feature policy differs from frozen DEV task')
    for e,meta in refs:
        with np.load(HERE/e['npz'],allow_pickle=False) as data:
            if not np.array_equal(data['train_uncapped'],full[0]) or not np.array_equal(data['val'],full[1]) or not np.array_equal(data['test'],full[2]):
                raise ValueError('Chronological boundary differs from frozen DEV task')
    pending.mkdir();(pending/'data').mkdir()
    entries=[]
    freeze_hash=sha(FREEZE)
    for seed in SEEDS:
        train=np.sort(np.random.default_rng(np.random.SeedSequence([seed,15002000])).choice(full[0],600,replace=False))
        val,test=full[1:]
        Z,target,pre=prior.standardize(X,y,train)
        assert groups[train].max()==2012 and set(groups[val])=={2013} and set(groups[test])=={2014,2015}
        arrays=dict(X=np.asarray(Z,np.float32),train=np.asarray(train,np.int64),val=np.asarray(val,np.int64),test=np.asarray(test,np.int64),
                    task_type=np.asarray('regression'),groups=np.asarray(groups),train_uncapped=np.asarray(full[0],np.int64),
                    y_train=np.asarray(target[train],np.float64),y_val=np.asarray(target[val],np.float64),y_test=np.asarray(target[test],np.float64))
        assert np.isfinite(arrays['X']).all() and np.isfinite(arrays['y_train']).all() and np.isfinite(arrays['y_val']).all()
        name=f'{CASE}__s{seed}';path=pending/'data'/f'{name}.npz'
        np.savez_compressed(path,**arrays)
        meta=copy.deepcopy(reference)
        meta.update(dataset=name,seed=seed,phase='confirmation',protocol_version='advantage-real-confirm-v1-20261028',preprocessing=pre,
                    source=source,split_protocol=protocol,npz_sha256=sha(path),array_sha256={k:dev.ahash(v) for k,v in arrays.items()},
                    confirmation_freeze_sha256=freeze_hash,confirmation_frozen_utc=doc['frozen_utc'],
                    frozen_choices=doc['choices'][CASE],
                    outcome_selection='All two-case/six-method parameters and hashes frozen before materializing any reserved confirmation seed. No fit or evaluation performed by this generator.')
        dev.dump(pending/'data'/f'{name}.json',meta)
        entries.append(dict(dataset=name,case=CASE,kind='real',family=CASE,phase='confirmation',seed=seed,task_type='regression',
                            npz=f'data/{name}.npz',metadata=f'data/{name}.json',npz_sha256=meta['npz_sha256'],source_novelty=source['source_novelty'],
                            rows=len(y),features=X.shape[1],split_sizes=meta['split_sizes']))
    dev.dump(pending/'manifest.json',dict(version='advantage-real-confirm-v1-20261028',entries=entries,
             confirmation_freeze_sha256=freeze_hash,confirmation_frozen_utc=doc['frozen_utc'],confirmation_seeds=SEEDS,
             development_manifest_sha256=sha(HERE/'manifest.json'),development_generator_sha256=sha(HERE/'prepare_real.py'),
             generator_sha256=sha(__file__),frozen_choices=doc['choices'][CASE],model_fits=0,test_evaluations=0,
             interpretation='Five training subsamples share fixed 2013 validation and 2014–15 test. NOX is a correlated target from the previously tested CO source, not five independent datasets.'))
    if sha(FREEZE)!=freeze_hash:raise ValueError('Freeze changed during generation; pending output not published')
    pending.rename(output)
    return output/'manifest.json'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generate',action='store_true',help='Materialize five packages after validating the complete root freeze; default is check-only')
    args=p.parse_args();doc,dev,manifest,refs=check_inputs()
    if args.generate:print('Generated: '+str(generate(doc,dev,manifest,refs)))
    else:print('PASS: complete root freeze and unchanged DEV source/protocol; no confirmation data generated')

if __name__=='__main__':main()
