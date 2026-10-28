#!/usr/bin/env python3
"""Versioned, portable fastlane data preparation; no model or outcome selection."""
from __future__ import annotations
import argparse, hashlib, io, json, re, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 'fastlane-v2-20261027-fixed-real-test'
DEV = [97001, 97002]
CONFIRM = list(range(98001, 98006))
UCI = {
 'gas_turbine_co': (551, 'gas+turbine+co+and+nox+emission+data+set'),
 'parkinson_total': (189, 'parkinsons+telemonitoring'),
 'superconductivity': (464, 'superconductivty+data'),
}

def digest(x):
    return hashlib.sha256(x if isinstance(x, bytes) else Path(x).read_bytes()).hexdigest()

def ahash(x):
    x = np.ascontiguousarray(x)
    return digest(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes())

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def random_split(n, seed, groups=None):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 6012020]))
    units = np.arange(n) if groups is None else np.unique(groups)
    units = rng.permutation(units)
    a, b = int(.6 * len(units)), int(.8 * len(units))
    if groups is None:
        return tuple(np.sort(z).astype(np.int64) for z in (units[:a], units[a:b], units[b:]))
    return tuple(np.flatnonzero(np.isin(groups, z)) for z in (units[:a], units[a:b], units[b:]))

def frozen_test_split(n, seed, groups=None):
    """One untouched real-data test set; repeated train/val draws use its complement."""
    units=np.arange(n) if groups is None else np.unique(groups)
    outer=np.random.default_rng(np.random.SeedSequence([96900,6012020])).permutation(units)
    boundary=int(.8*len(outer))
    test_units=outer[boundary:]
    remaining=np.random.default_rng(np.random.SeedSequence([seed,7525])).permutation(outer[:boundary])
    cut=int(.75*len(remaining))
    parts=(remaining[:cut],remaining[cut:],test_units)
    if groups is None:
        return tuple(np.sort(v).astype(np.int64) for v in parts)
    return tuple(np.flatnonzero(np.isin(groups,v)) for v in parts)

def save(out, name, X, y, splits, meta, groups=None, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    arrays = dict(X=np.asarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.float64),
                  train=np.asarray(splits[0], dtype=np.int64), val=np.asarray(splits[1], dtype=np.int64),
                  test=np.asarray(splits[2], dtype=np.int64), task_type=np.asarray(meta.get('task_type', 'regression')))
    if groups is not None:
        arrays['groups'] = np.asarray(groups)
    if extra:
        arrays.update(extra)
    for split in ('train','val','test'):
        arrays['y_'+split]=arrays['y'][arrays[split]]
    assert np.isfinite(arrays['X']).all() and np.isfinite(arrays['y']).all()
    used = np.concatenate(splits)
    assert len(used) == len(np.unique(used))
    if groups is not None:
        sets = [set(np.asarray(groups)[z].tolist()) for z in splits]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    path = out / (name + '.npz')
    np.savez_compressed(path, **arrays)
    meta.update(protocol_version=VERSION, dataset=name, rows=len(y), features=X.shape[1],
                npz_sha256=digest(path), array_sha256={k: ahash(v) for k, v in arrays.items()},
                split_sizes={k: len(v) for k, v in zip(('train', 'val', 'test'), splits)},
                groups_in_X=False, oracle_support_in_X=False)
    if groups is not None:
        meta['group_counts'] = {k: len(np.unique(groups[z])) for k, z in zip(('train','val','test'), splits)}
    dump(out / (name + '.json'), meta)
    return dict(dataset=name, npz=path.name, metadata=name+'.json', npz_sha256=meta['npz_sha256'],
                phase=meta.get('phase'), seed=meta['seed'], task_type=meta.get('task_type','regression'))

def check_seeds(root):
    """Check integer seed use, not decimal substrings in old metrics."""
    collisions = []
    folder = root / 'rebuttal/preexperiments_20261027'
    for p in folder.rglob('*'):
        if p.suffix not in ('.py','.json','.md','.csv') or not p.is_file():
            continue
        if p.suffix == '.csv':
            try:
                frame = pd.read_csv(p)
                for c in frame.columns:
                    if 'seed' in c.lower():
                        found = set(pd.to_numeric(frame[c], errors='coerce').dropna().astype(int)) & set(DEV+CONFIRM)
                        if found: collisions.append({'file': str(p.relative_to(root)), 'seeds': sorted(found)})
            except (ValueError, pd.errors.ParserError, UnicodeError):
                pass
        else:
            found = [s for s in DEV+CONFIRM if re.search(r'(?<![\d.])'+str(s)+r'(?![\d.])', p.read_text(errors='replace'))]
            if found: collisions.append({'file': str(p.relative_to(root)), 'seeds': found})
    if collisions:
        raise RuntimeError('Seed reuse detected: ' + json.dumps(collisions))
    return {'searched': str(folder.relative_to(root)), 'seeds': DEV+CONFIRM, 'collisions': []}

def synthetic(out, seeds, noise, root):
    audit = check_seeds(root)
    entries=[]
    for family,n,d in [('S1',240,32),('S2',480,32),('S3',480,16)]:
        for seed in seeds:
            rng=np.random.default_rng(np.random.SeedSequence([seed, int(family[1:]), 100]))
            X=rng.normal(size=(n,d))
            eps=rng.normal(scale=noise,size=n)
            sign=np.where(X[:,0]>=0,1.,-1.)
            if family=='S1':
                region=(X[:,0]>=0).astype(int)
                sparse=2*sign*X[:,1]
                dense=2*sign*X[:,1:].sum(axis=1)/np.sqrt(d-1)
                offset=2*sign
                formula='2 sign(x0) x1 + eps'
                support={'gate_features':[0], 'predictive_by_region':{'0':[1],'1':[1]}, 'slopes':[-2,2]}
            elif family=='S2':
                region=2*(X[:,0]>=0).astype(int)+(X[:,1]>=0).astype(int)
                slopes=np.array([-2.,2.,-2.,2.])
                sparse=slopes[region]*X[np.arange(n),2+region]
                # Four disjoint dense modules, each with equal total slope norm 2.
                blocks=np.array_split(np.arange(2,d),4)
                dense=np.zeros(n)
                for k,b in enumerate(blocks):
                    mask=region==k
                    dense[mask]=slopes[k]*X[mask][:,b].sum(axis=1)/np.sqrt(len(b))
                offset=slopes[region]
                formula='[-2,+2,-2,+2][region] * x[2+region] + eps; region=2 I(x0>=0)+I(x1>=0)'
                support={'gate_features':[0,1], 'predictive_by_region':{str(k):[2+k] for k in range(4)}, 'slopes':slopes.tolist()}
            else:
                sign2=np.where(X[:,1]>=0,1.,-1.)
                region=2*(X[:,0]>=0).astype(int)+(X[:,1]>=0).astype(int)
                sparse=2/np.sqrt(2)*(sign*X[:,2]+sign2*X[:,3])
                b1,b2=np.array_split(np.arange(2,d),2)
                dense=2/np.sqrt(2)*(sign*X[:,b1].sum(axis=1)/np.sqrt(len(b1))+sign2*X[:,b2].sum(axis=1)/np.sqrt(len(b2)))
                offset=2/np.sqrt(2)*(sign+sign2)
                formula='2/sqrt(2) [sign(x0) x2 + sign(x1) x3] + eps'
                support={'gate_features':[0,1], 'predictive_modules':[[2],[3]], 'per_module_slope_magnitude':float(2/np.sqrt(2))}
            splits=random_split(n,seed)
            for control,signal in [('sparse',sparse),('dense',dense),('offset',offset)]:
                name=f'{family}_{control}__s{seed}'
                meta=dict(seed=seed, phase='development' if seed in DEV else 'confirmation', task_type='regression',
                          family=family,control=control,source='generated independent N(0,1) covariates',
                          noise_sd=noise,formula=formula if control=='sparse' else control+' control; see data_protocol.md',
                          oracle_support=support if control=='sparse' else None,
                          feature_names=[f'x{i}' for i in range(d)],preprocessing={'X':'none','y':'none'},
                          matching={'X_sha256':ahash(X),'eps_sha256':ahash(eps),'population_signal_variance':4.0},
                          split_protocol='seeded row random 60/20/20; shared across controls',seed_collision_check=audit)
                # Oracle region/signal remain diagnostic arrays and are never concatenated to X.
                entries.append(save(out,name,X,signal+eps,splits,meta,extra={'oracle_region':region,'oracle_signal':signal,'noise':eps}))
    return entries

def fetch_uci(name,raw,download):
    ident,slug=UCI[name]
    raw.mkdir(parents=True,exist_ok=True)
    archive=raw/f'uci_{ident}.zip'
    url=f'https://archive.ics.uci.edu/static/public/{ident}/{slug.replace("+","%2B")}.zip'
    if not archive.exists():
        if not download: raise FileNotFoundError(f'{archive}; use --download for official public archive')
        with urllib.request.urlopen(url,timeout=60) as r:
            content=r.read()
        zipfile.ZipFile(io.BytesIO(content))
        archive.write_bytes(content)
    return zipfile.ZipFile(archive), dict(uci_id=ident,source_url=f'https://archive.ics.uci.edu/dataset/{ident}/{slug}',archive_url=url,archive_sha256=digest(archive))

def real_frame(name,root,raw,download):
    local={
     'cpu_activity':'check_codereview0701/tabular_extra_random300/data/regression__cpu_activity.csv',
     'protein':'check_codereview0701/dataset/TabArena_regression/physiochemical_protein.csv',
     'kin8nm':'check_codereview0701/tabular_extra_random300/data/regression__kin8nm.csv',
    }
    if name in local:
        path=root/local[name]
        df=pd.read_csv(path)
        return df.iloc[:,:-1], df.iloc[:,-1], None, None, dict(source_file=local[name],source_sha256=digest(path),
                    provenance='simulation' if name=='kin8nm' else 'observational', target=df.columns[-1])
    z,source=fetch_uci(name,raw,download)
    def read(suffix):
        matches=[v for v in z.namelist() if v.endswith(suffix)]
        if len(matches)!=1: raise ValueError((suffix,matches))
        return pd.read_csv(io.BytesIO(z.read(matches[0])))
    if name=='gas_turbine_co':
        frames=[read(f'gt_{year}.csv').assign(year=year) for year in range(2011,2016)]
        df=pd.concat(frames,ignore_index=True)
        years=df['year'].to_numpy()
        splits=tuple(np.flatnonzero(np.isin(years,v)) for v in ([2011,2012],[2013],[2014,2015]))
        source.update(target='CO',excluded_features=['CO','NOX','year'],provenance='sensor measurements',
                      year_counts={str(y):int(sum(years==y)) for y in np.unique(years)})
        return df.drop(columns=['CO','NOX','year']),df.CO,None,splits,source
    if name=='parkinson_total':
        df=read('parkinsons_updrs.data')
        voice=list(df.columns[6:])
        assert len(voice)==16 and not any('UPDRS' in c for c in voice)
        source.update(target='total_UPDRS',excluded_features=list(df.columns[:6]),provenance='repeated biomedical voice measurements')
        return df[voice],df.total_UPDRS,df['subject#'].to_numpy(),None,source
    df=read('train.csv'); comp=read('unique_m.csv')
    assert len(df)==len(comp) and np.allclose(df.critical_temp,comp.critical_temp)
    # Elemental composition vector, normalized by total atom count: same formula
    # in different element order or stoichiometric multiples remains one group.
    elems=comp.drop(columns=['critical_temp','material'])
    values=elems.to_numpy(float); values/=values.sum(axis=1,keepdims=True)
    keys=[','.join(f'{v:.12g}' for v in row) for row in values]
    groups=np.unique(keys,return_inverse=True)[1]
    source.update(target='critical_temp',excluded_features=['material','critical_temp (unique_m)'],
                  composition_alignment_verified=True,composition_grouping='normalized full elemental composition, 12 significant digits',
                  composition_groups=int(len(np.unique(groups))),material_strings=int(comp.material.nunique()),provenance='materials measurements')
    return df.drop(columns='critical_temp'),df.critical_temp,groups,None,source

def standardize(X,y,train):
    values=np.asarray(X,dtype=float)
    values[~np.isfinite(values)]=np.nan
    med=np.nanmedian(values[train],axis=0);med[~np.isfinite(med)]=0
    values=np.where(np.isnan(values),med,values)
    mean=values[train].mean(axis=0);scale=values[train].std(axis=0);scale[scale<1e-12]=1
    ym=float(np.mean(y[train]));ys=float(np.std(y[train]));ys=max(ys,1e-12)
    return (values-mean)/scale,(y-ym)/ys,dict(fit_rows_sha256=ahash(train),X_imputation_median=med.tolist(),X_mean=mean.tolist(),X_scale=scale.tolist(),y_mean=ym,y_scale=ys,
             fit_scope='capped training rows only',centering_note='X/y centering can change slope-only leaf model behavior; identical transform is supplied to every method')

def real(out,names,seeds,root,raw,download,cap):
    entries=[]
    for name in names:
        X,y,groups,fixed,source=real_frame(name,root,raw,download)
        y=np.asarray(y,dtype=float)
        assert np.isfinite(y).all()
        for seed in seeds:
            full=fixed or frozen_test_split(len(y),seed,groups)
            train,val,test=full
            if cap and len(train)>cap:
                train=np.sort(np.random.default_rng(np.random.SeedSequence([seed,15002000])).choice(train,cap,replace=False))
            transformed,target,pre=standardize(X,y,train)
            phase='development' if seed in DEV else 'confirmation'
            meta=dict(seed=seed,phase=phase,task_type='regression',source=source,feature_names=list(X.columns),preprocessing=pre,
                 split_protocol='2011-12 / 2013 / 2014-15 chronological' if fixed else ('fixed outer 20% test by subject/composition, seed96900; repeated 75/25 train/val within remaining80%' if groups is not None else 'fixed outer20% row test, seed96900; repeated75/25 train/val within remaining80%'),
                 outer_test_seed=None if fixed else 96900,
                 repeat_interpretation='Development and confirmation train/val repeats share one held-out test set; not independent new datasets.',
                 training_cap=cap,training_cap_rule='uniform index sampling without labels; val/test unchanged',
                 uncapped_split_sizes={k:len(z) for k,z in zip(('train','val','test'),full)},
                 uncapped_target_metadata={'train':{'n':len(full[0]),'sha256':ahash(y[full[0]])}},
                 note='No validation/test label summaries are emitted. Test labels must remain unavailable during configuration selection.')
            entries.append(save(out,f'{name}__s{seed}',transformed,target,(train,val,test),meta,groups,
                                extra={'train_uncapped':full[0]}))
    return entries

def cpu_stats(path,out):
    df=pd.read_csv(path); X=df.iloc[:,:-1].apply(pd.to_numeric); y=df.iloc[:,-1]
    corr=X.corr(); upper=np.triu(np.ones(corr.shape,dtype=bool),1)
    pairs=[{'a':corr.index[i],'b':corr.columns[j],'correlation':float(corr.iat[i,j])} for i,j in zip(*np.where(upper & (corr.abs().to_numpy()>=.98)))]
    finite=corr.abs().where(upper).stack()
    feature_stats={c:{'min':float(X[c].min()),'max':float(X[c].max()),'mean':float(X[c].mean()),'std':float(X[c].std(ddof=0)),'skew':float(X[c].skew()) if X[c].nunique()>1 else 0.,'unique':int(X[c].nunique()),'raw_second_moment':float(np.mean(X[c]**2))} for c in X}
    # Equal h=1, single-coordinate denominators sum(x_j^2) are only a data
    # geometry proxy. Real leaves have selected samples, evolving h, and ridge.
    raw_m2=(X**2).mean().to_numpy()
    Z=(X-X.mean())/X.std(ddof=0).replace(0,1)
    norm2=(Z**2).sum(axis=1).to_numpy()
    result=dict(source=str(path),source_sha256=digest(path),rows=len(df),features=len(X.columns),target=df.columns[-1],constants=[c for c in X if X[c].nunique()<=1],feature_stats=feature_stats,
                max_absolute_feature_correlation={'pair':list(finite.idxmax()),'value':float(finite.max())},near_duplicate_correlation_pairs=pairs,
                exact_duplicate_feature_pairs=[{'a':a,'b':b} for i,a in enumerate(X) for b in list(X)[i+1:] if X[a].equals(X[b])],
                target_range=[float(y.min()),float(y.max())],
                scalar_denominator_proxies={'assumption':'h=1, whole table, no ridge; descriptive only',
                   'raw_feature_second_moment_max_over_min':float(raw_m2.max()/raw_m2[raw_m2>0].min()),
                   'globally_standardized_row_norm_squared_quantiles':{str(q):float(np.quantile(norm2,q)) for q in [0,.5,.9,.99,1]},
                   'globally_standardized_row_norm_squared_cv':float(norm2.std()/norm2.mean()),
                   'standardization_scope':'whole table used for descriptive statistics only; model inputs use training rows only'},
                interpretation='Descriptive scalar denominator/scale proxies only. Correlation and scale variation do not identify a causal normalization mechanism.')
    dump(out,result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--out',type=Path)
    p.add_argument('--raw',type=Path)
    p.add_argument('--mode',choices=['synthetic','real','all','cpu-stats'],default='synthetic')
    p.add_argument('--seeds',type=int,nargs='+',default=DEV+CONFIRM)
    p.add_argument('--noise',type=float,default=.3)
    p.add_argument('--datasets',nargs='+',default=['cpu_activity','protein','kin8nm','gas_turbine_co','parkinson_total','superconductivity'])
    p.add_argument('--train-cap',type=int,default=1800)
    p.add_argument('--download',action='store_true')
    args=p.parse_args();root=args.root.resolve();out=args.out or root/'local_sync/fastlane_data';raw=args.raw or out/'raw'
    entries=[]
    if args.mode in ('synthetic','all'):entries+=synthetic(out,args.seeds,args.noise,root)
    if args.mode in ('real','all'):entries+=real(out,args.datasets,args.seeds,root,raw,args.download,args.train_cap)
    if args.mode in ('cpu-stats','all'):
        cpu_stats(root/'check_codereview0701/tabular_extra_random300/data/regression__cpu_activity.csv',out/'cpu_activity_descriptive_stats.json')
    if entries:
        manifest=out/f'manifest_{args.mode}.json'
        dump(manifest,{'protocol_version':VERSION,'development_seeds':DEV,'confirmation_seeds':CONFIRM,'datasets':entries})
        print(json.dumps({'manifest':str(manifest),'datasets':len(entries)}))
    else:print(str(out/'cpu_activity_descriptive_stats.json'))

if __name__=='__main__':main()
