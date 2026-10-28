#!/usr/bin/env python3
"""Add train-only encoded ILPD/website panels without changing existing data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

SEEDS=(97001,97002,98001,98002,98003,98004,98005)
SOURCES={
 'ilpd_numeric':'check_codereview0701/tabular_extra_random300/data/binary__ilpd.csv',
 'website_phishing':'check_codereview0701/dataset/TabArena_multiclass/website_phishing.csv',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ahash(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()

def transform(raw,train):
    numeric=list(raw.select_dtypes(include=[np.number]).columns)
    categorical=[c for c in raw if c not in numeric]
    chunks=[];names=[];meta={'fit_scope':'training rows only','numeric_columns':numeric,'categorical_columns':categorical}
    if numeric:
        values=raw[numeric].to_numpy(float);values[~np.isfinite(values)]=np.nan
        median=np.nanmedian(values[train],axis=0);median[~np.isfinite(median)]=0
        values=np.where(np.isnan(values),median,values)
        mean=values[train].mean(0);scale=values[train].std(0);scale[scale<1e-12]=1
        chunks.append((values-mean)/scale);names+=numeric
        meta.update(numeric_median=median.tolist(),numeric_mean=mean.tolist(),numeric_scale=scale.tolist())
    if categorical:
        values=raw[categorical].astype('string').fillna('MISSING').astype(str)
        encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        encoder.fit(values.iloc[train])
        chunks.append(encoder.transform(values));names+=encoder.get_feature_names_out(categorical).tolist()
        meta.update(categorical_imputation='literal MISSING',onehot_categories=[v.tolist() for v in encoder.categories_],unknown_categories='all-zero encoding; handle_unknown=ignore')
    meta.update(y='train-fitted LabelEncoder; no centering or scaling',y_mean=0.,y_scale=1.,
                centering_note='numeric feature centering affects slope-only leaves; identical inputs are supplied to all methods')
    return np.column_stack(chunks).astype(np.float32),names,meta

def prepare(root,data):
    targets=[data/f'{name}__s{seed}{suffix}' for name in SOURCES for seed in SEEDS for suffix in ('.npz','.json')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Refusing to overwrite existing classification output')
    data.mkdir(parents=True,exist_ok=True)
    report=[]
    for name,relative in SOURCES.items():
        path=root/relative;frame=pd.read_csv(path)
        raw=frame.iloc[:,:-1].copy();labels=frame.iloc[:,-1].copy()
        assert labels.notna().all()
        groups=None;group_meta={}
        dropped=[]
        if name=='ilpd_numeric':
            dropped=[c for c in raw if c not in raw.select_dtypes(include=[np.number]).columns]
            raw=raw.select_dtypes(include=[np.number]).copy()
            assert dropped==['V2'] and len(raw.columns)==9
            pool,test=train_test_split(np.arange(len(labels)),test_size=.2,random_state=96900,stratify=labels)
            limitations=['583 rows; minority class has167 rows before splitting; variance can be large.',
                'Numeric-only drop mode excludes gender V2, preserving the existing primary preprocessing estimand.',
                'Row-stratified split, not patient-group validation; patient identifiers are unavailable. Exact duplicate numeric inputs may occur across partitions.',
                'All repeats share a fixed test set; repeats are not independent datasets.']
        else:
            # Exact raw feature tuples only. Labels never enter group identity.
            groups=pd.factorize(pd.MultiIndex.from_frame(raw),sort=False)[0].astype(np.int64)
            outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=96900)
            pool,test=next(outer.split(raw,labels,groups))
            sizes=pd.Series(groups).value_counts()
            conflicts=pd.DataFrame({'group':groups,'label':labels}).groupby('group')['label'].nunique()
            group_meta=dict(definition='exact tuple of all9 raw feature columns; target excluded',
                            total_groups=int(len(sizes)),duplicate_feature_rows=int(len(raw)-len(sizes)),
                            largest_group_rows=int(sizes.max()),conflicting_label_groups=int((conflicts>1).sum()),
                            class_group_support={str(label):int(len(np.unique(groups[labels.to_numpy()==label]))) for label in labels.unique()})
            limitations=['Approximate class stratification constrained by indivisible feature groups; row fractions need not be exactly60/20/20.',
                '1353 rows, with only103 Suspicious-class rows before splitting; group-level support and uncertainty limit conclusions.',
                'Exact-feature grouping measures generalization to unseen feature combinations and changes the IID-row estimand.',
                'Identical-feature groups can contain conflicting labels; groups remain intact.',
                'All repeats share a fixed group-held-out test set; repeats are not independent datasets.']
        test=np.sort(test)
        for seed in SEEDS:
            if groups is None:
                train,val=train_test_split(pool,test_size=.25,random_state=seed,stratify=labels.iloc[pool])
            else:
                splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed)
                tr,va=next(splitter.split(raw.iloc[pool],labels.iloc[pool],groups[pool]));train,val=pool[tr],pool[va]
            splits={k:np.sort(v).astype(np.int64) for k,v in [('train',train),('val',val),('test',test)]}
            assert len(np.unique(np.concatenate(list(splits.values()))))==len(labels)
            if groups is not None:
                sets=[set(groups[v]) for v in splits.values()]
                assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            X,features,pre=transform(raw,splits['train'])
            encoder=LabelEncoder().fit(labels.iloc[splits['train']])
            y=encoder.transform(labels).astype(np.int64)
            for split,indices in splits.items():
                assert set(y[indices])==set(range(len(encoder.classes_))),f'{name}/{seed}/{split}:class missing'
            task='binary' if len(encoder.classes_)==2 else 'multiclass'
            arrays=dict(X=X,y=y,task_type=np.asarray(task),**splits)
            arrays.update({'y_'+k:y[v] for k,v in splits.items()})
            if groups is not None:arrays['groups']=groups
            assert np.isfinite(X).all()
            stem=f'{name}__s{seed}';output=data/(stem+'.npz')
            with output.open('xb') as stream:np.savez_compressed(stream,**arrays)
            meta=dict(protocol_version='fastlane-v2-20261027-classification',dataset=stem,task_type=task,seed=seed,
                phase='development' if seed<98000 else 'confirmation',rows=len(y),features=X.shape[1],feature_names=features,
                source_file=relative,source_sha256=sha(path),target_column=frame.columns[-1],
                dropped_features=dropped,drop_reason='gender V2 excluded by numeric-only primary mode' if dropped else None,
                class_encoding={'fit_scope':'training rows only','classes':encoder.classes_.tolist()},
                preprocessing=pre,npz_sha256=sha(output),array_sha256={k:ahash(v) for k,v in arrays.items()},
                split_sizes={k:len(v) for k,v in splits.items()},outer_test_seed=96900,
                split_protocol='fixed stratified outer20%; seed-varying stratified75/25 train/val in complement' if groups is None else 'fixed outer firstfold StratifiedGroupKFold5 seed96900; seed-varying firstfold StratifiedGroupKFold4 on complement',
                groups_in_X=False,oracle_support_in_X=False,grouping=group_meta,
                group_counts={k:len(np.unique(groups[v])) for k,v in splits.items()} if groups is not None else None,
                test_overlap_with_train_or_val=0,
                repeat_interpretation='one fixed held-out real test set; seven train/validation repeats are not independent datasets',
                limitations=limitations,model_fit_calls=0)
            with output.with_suffix('.json').open('x') as stream:stream.write(json.dumps(meta,indent=2,allow_nan=False)+'\n')
            report.append(dict(dataset=stem,npz=str(output),sha256=meta['npz_sha256'],task_type=task,
                features=X.shape[1],split_sizes=meta['split_sizes'],test_index_hash=meta['array_sha256']['test'],grouping=group_meta))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--data',type=Path)
    args=p.parse_args();root=args.root.resolve();data=args.data or root/'local_sync/fastlane_data'
    print(json.dumps({'created':prepare(root,data.resolve()),'model_fit_calls':0,'existing_files_modified':0},indent=2))

if __name__=='__main__':main()
