#!/usr/bin/env python3
"""Prepare four predeclared real regression tasks; never fit or score a model.

Only DEV packages are materialized. CONFIRM seeds are reserved, not evaluated.
Source archives are downloaded separately from official UCI endpoints.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, io, json, re, warnings, zipfile
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEV = [112001, 112002]
CONFIRM = list(range(113001, 113006))
SPECS = {
    'residential_cost': (437, 'residential+building+data+set', '10.24432/C5S896'),
    'communities_crime': (183, 'communities+and+crime', '10.24432/C53W3X'),
    'parkinson_motor': (189, 'parkinsons+telemonitoring', '10.24432/C5ZS3N'),
    'gas_turbine_nox': (551, 'gas+turbine+co+and+nox+emission+data+set', '10.24432/C5WC95'),
}

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ahash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def prior_runtime():
    path = ROOT/'rebuttal/fastlane_20261027/prepare_data.py'
    spec=importlib.util.spec_from_file_location('prior_prepare', path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def seed_audit():
    collisions=[]
    scanned=[]
    for folder in ['rebuttal/preexperiments_20261027','rebuttal/fastlane_20261027','rebuttal/fastlane_20261028_bc']:
        scanned.append(folder)
        for p in (ROOT/folder).rglob('*'):
            if p.suffix not in ['.json','.py','.md','.csv'] or not p.is_file():continue
            text=p.read_text(errors='replace')
            hits=[s for s in DEV+CONFIRM if re.search(r'(?<![\d.])'+str(s)+r'(?![\d.])',text)]
            if hits:collisions.append({'file':str(p.relative_to(ROOT)),'seeds':hits})
    assert not collisions, collisions
    return dict(searched=scanned, seeds=DEV+CONFIRM, collisions=[])

def chronology_blocks(groups, fraction=.2):
    units=np.unique(groups)
    # Choose nearest row fraction while retaining whole contemporaneous blocks.
    cut=min(range(1,len(units)),key=lambda k:abs(np.sum(groups>=units[k])-fraction*len(groups)))
    return np.flatnonzero(groups<units[cut]),np.flatnonzero(groups>=units[cut])

def load_task(name, prior):
    ident,slug,doi=SPECS[name]
    archive=HERE/'sources'/f'uci_{ident}.zip'
    z=zipfile.ZipFile(archive)
    def read(suffix,**kwargs):
        names=[n for n in z.namelist() if n.endswith(suffix)]
        assert len(names)==1
        return pd.read_csv(io.BytesIO(z.read(names[0])),**kwargs)
    src=dict(uci_id=ident,source_url=f'https://archive.ics.uci.edu/dataset/{ident}/{slug}',
             archive_url=f'https://archive.ics.uci.edu/static/public/{ident}/{slug.replace("+","%2B")}.zip',
             archive_sha256=digest(archive),doi=doi,license='CC BY 4.0',
             license_url='https://creativecommons.org/licenses/by/4.0/',
             official_metadata_file=f'sources/uci_{ident}_official_metadata.json',
             official_metadata_sha256=digest(HERE/'sources'/f'uci_{ident}_official_metadata.json'))
    if name=='residential_cost':
        raw=pd.read_excel(io.BytesIO(z.read('Residential-Building-Data-Set.xlsx')),sheet_name='Data',header=None)
        df=raw.iloc[2:].reset_index(drop=True).apply(pd.to_numeric)
        X=df.iloc[:,4:107].copy()
        X.columns=[f'V-{i}' for i in range(1,9)]+[f'V-{i}_lag{lag}' for lag in range(1,6) for i in range(11,30)]
        y=df.iloc[:,108].to_numpy(float)
        groups=(df.iloc[:,2].to_numpy(int)*4+df.iloc[:,3].to_numpy(int)-1)
        pool,test=chronology_blocks(groups)
        it,iv=chronology_blocks(groups[pool],.25)
        fixed=(pool[it],pool[iv],test)
        src.update(target='V-10 actual construction costs',excluded_features=['V-9 actual sale prices','four project date metadata columns'],
                   provenance='observed residential construction projects',source_novelty='new_source_in_examined_local_runs',
                   task_definition='Completed-project retrospective cost estimation; not a project-start forecast.',
                   leakage_audit='Descriptions: 19 economic variables at 5 lags before project start; actual construction duration V-7 is retained and only valid for completed-project estimation.',
                   historical_evidence='Absent from preexperiments 52-source inventory and expansion_01/02 and fastlane preparation registry.',
                   upstream_preprocessing='Economic indices are published with calibration/normalization; only train-fitted transformations added here.')
        protocol='Chronological completion-quarter blocks, nearest 20% outer test; preceding pool split 75/25 by whole completion quarters. Same fixed boundary for both DEV seeds.'
    elif name=='communities_crime':
        names=re.findall(r'^@attribute\s+(\S+)\s+',z.read('communities.names').decode(),flags=re.M|re.I)
        df=read('communities.data',header=None,names=names,na_values='?')
        assert df.shape==(1994,128)
        groups=df.state.to_numpy(int)
        X=df.iloc[:,5:-1].apply(pd.to_numeric);y=df.iloc[:,-1].to_numpy(float)
        fixed=None
        src.update(target='ViolentCrimesPerPop',excluded_features=names[:5],provenance='observational census, law-enforcement and crime records',
                   source_novelty='new_source_in_examined_local_runs',
                   historical_evidence='Absent from preexperiments 52-source inventory and expansion_01/02 and fastlane preparation registry.',
                   upstream_preprocessing='UCI source was globally binned, normalized and clipped before release. This cannot be undone; benchmark uses released predictors and target, not raw population records.')
        protocol='Fixed outer 20% of state groups, seed96900; DEV train/val 75/25 state groups within complement. Percentages refer to groups, not exact row counts.'
    elif name=='parkinson_motor':
        df=read('parkinsons_updrs.data');X=df.iloc[:,6:].copy();y=df.motor_UPDRS.to_numpy(float)
        groups=df['subject#'].to_numpy(int);fixed=None
        assert len(X.columns)==16
        src.update(target='motor_UPDRS',excluded_features=list(df.columns[:6]),provenance='repeated biomedical voice measurements',
                   source_novelty='existing_source_new_target',
                   historical_evidence='Same UCI189 source and subject grouping already tested in fastlane with total_UPDRS. This correlated motor target is not an independent new dataset.',
                   leakage_audit='Both UPDRS columns, subject ID, age, sex and test_time excluded to preserve prior voice-only task. Subject ID stored only as groups.',
                   target_measurement_limitation='Clinical motor scores are linearly interpolated between visits in the released source; recordings are real, labels are not independent per recording.')
        protocol='Reuse prior fixed outer 20% subject groups, seed96900; DEV train/val 75/25 subjects within complement. Percentages refer to subjects, not exact row counts.'
    else:
        df=pd.concat([read(f'gt_{year}.csv').assign(year=year) for year in range(2011,2016)],ignore_index=True)
        groups=df.year.to_numpy(int);X=df.drop(columns=['CO','NOX','year']);y=df.NOX.to_numpy(float)
        fixed=tuple(np.flatnonzero(np.isin(groups,years)) for years in ([2011,2012],[2013],[2014,2015]))
        src.update(target='NOX',excluded_features=['CO','NOX','year'],provenance='real hourly gas-turbine sensor measurements',
                   source_novelty='existing_source_new_target',
                   historical_evidence='Same UCI551 source already tested in fastlane/formal with CO. NOX is a correlated new target, not an independent new source; same plant as UCI294 CCPP with another measurement period.',
                   leakage_audit='Other flue-gas output CO excluded; all nine ambient/operating sensors retained.',
                   chronology='Official UCI description confirms rows are chronological within year; original calendar timestamps are not released.')
        protocol='Preserve established chronological 2011–12 train / 2013 validation / 2014–15 test. Outer test is about40%, deliberately not changed to20% after historical outcomes.'
    assert X.shape[1] in [103,122,16,9]
    # Diagnostics use X and metadata only; targets never determine task/splits.
    src['duplicate_predictor_rows']=int(X.duplicated().sum())
    return X,y,groups,fixed,src,protocol

def save_task(name,seed,prior,audit):
    X,y,groups,fixed,source,protocol=load_task(name,prior)
    full=fixed if fixed is not None else prior.frozen_test_split(len(y),seed,groups)
    train,val,test=full
    if len(train)>600:
        train=np.sort(np.random.default_rng(np.random.SeedSequence([seed,15002000])).choice(train,600,replace=False))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        Z,target,pre=prior.standardize(X,y,train)
    splits=(train,val,test)
    assert len(np.unique(np.concatenate(splits)))==sum(map(len,splits))
    gs=[set(groups[i].tolist()) for i in splits]
    assert not(gs[0]&gs[1] or gs[0]&gs[2] or gs[1]&gs[2])
    if fixed is not None:
        assert groups[train].max()<groups[val].min()<groups[test].min()
    arrays=dict(X=np.asarray(Z,np.float32),train=np.asarray(train,np.int64),val=np.asarray(val,np.int64),test=np.asarray(test,np.int64),
                task_type=np.asarray('regression'),groups=np.asarray(groups),train_uncapped=np.asarray(full[0],np.int64),
                y_train=np.asarray(target[train],np.float64),y_val=np.asarray(target[val],np.float64),y_test=np.asarray(target[test],np.float64))
    assert np.isfinite(arrays['X']).all()
    assert np.isfinite(arrays['y_train']).all() and np.isfinite(arrays['y_val']).all()
    out=HERE/'data';out.mkdir(exist_ok=True)
    dataset=f'{name}__s{seed}';path=out/f'{dataset}.npz'
    np.savez_compressed(path,**arrays)
    # Hash target bytes during sealing only. Verification/search never reads y_test.
    meta=dict(protocol_version='advantage-real-dev-v1-20261028',dataset=dataset,case=name,family=name,kind='real',phase='development',seed=seed,task_type='regression',
              source=source,feature_names=list(X.columns),rows=len(y),features=X.shape[1],preprocessing=pre,
              split_protocol=protocol,split_sizes={k:len(v) for k,v in zip(['train','val','test'],splits)},
              uncapped_split_sizes={k:len(v) for k,v in zip(['train','val','test'],full)},
              group_counts={k:len(np.unique(groups[v])) for k,v in zip(['train','val','test'],splits)},
              training_cap=600,training_cap_rule='Uniform row sample from training groups/period only, without target inspection; full train if <=600.',
              groups_in_X=False,oracle_support_in_X=False,oracle_support=None,npz_sha256=digest(path),
              array_sha256={k:ahash(v) for k,v in arrays.items()},seed_collision_check=audit,
              repeat_interpretation='Two DEV seeds and five reserved CONFIRM seeds share fixed source-specific outer tests; neither repeats nor correlated targets are independent datasets.',
              identical_data_across_dev_seeds=(name=='residential_cost'),
              test_label_policy='y_test sealed as separate NPZ member; no combined y member. No model fitting, scoring or test target summaries in preparation.',
              outcome_selection='Targets and feature policies fixed before any fit; no target correlation or model-result screening was used.')
    dump(out/f'{dataset}.json',meta)
    return dict(dataset=dataset,case=name,kind='real',family=name,phase='development',seed=seed,task_type='regression',npz=f'data/{dataset}.npz',metadata=f'data/{dataset}.json',
                npz_sha256=meta['npz_sha256'],source_novelty=source['source_novelty'],rows=len(y),features=X.shape[1],split_sizes=meta['split_sizes'])

def main():
    audit=seed_audit();prior=prior_runtime()
    entries=[save_task(name,seed,prior,audit) for name in SPECS for seed in DEV]
    manifest=dict(version='advantage-real-dev-v1-20261028',entries=entries,dev_seeds=DEV,reserved_confirmation_seeds=CONFIRM,
                  confirmation_materialized=False,model_fits=0,test_evaluations=0,independent_source_accounting='2 newly examined local sources + 2 correlated new targets from previously tested sources; not four new independent benchmarks.',
                  prior_generator_path='rebuttal/fastlane_20261027/prepare_data.py',prior_generator_sha256=digest(ROOT/'rebuttal/fastlane_20261027/prepare_data.py'),
                  generator_sha256=digest(__file__),seed_collision_check=audit)
    dump(HERE/'manifest.json',manifest)
    print(json.dumps([dict(case=e['case'],seed=e['seed'],features=e['features'],split_sizes=e['split_sizes']) for e in entries],indent=2))

if __name__=='__main__':main()
