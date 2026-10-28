#!/usr/bin/env python3
"""Freeze B1 reuse/jobs and B2/C inputs only; never import or execute a learner."""
from __future__ import annotations
import hashlib, importlib.util, json, math, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'inputs'
DEV = [97001, 97002]
NEW = list(range(99001, 99006))
METHODS = ['CrossGB','XGB','LGB_constant','LGB_linear','PL','LinXGBoost']
SOURCES = {
    'S1_sparse': 'cpu4_pool.zjuici.com',
    'S2_sparse': 'cpu2_pool.zjuici.com',
    'S3_sparse': 'cpu1_pool.zjuici.com',
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p): return str(Path(p).relative_to(ROOT))
def read(p): return json.loads(Path(p).read_text())
def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def source(case):
    return ROOT/'local_sync/fastlane_results'/SOURCES[case]/'results/formal24_v1'/case

def historical_seed_audit():
    # Strict integer boundaries avoid falsely reading seeds out of decimal metrics.
    cmd = ['rg','-l','--pcre2',r'(?<![\d.])9900[1-5](?![\d.])',
           'rebuttal','local_sync','--glob','*.json','--glob','*.csv',
           '--glob','*.py','--glob','*.md','--glob','!**/fastlane_20261028_bc/**']
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode not in (0,1): raise RuntimeError(result.stderr)
    matches = sorted(result.stdout.splitlines())
    proposal = 'rebuttal/报错修复与实验增量计划_20261028_待审阅.md'
    unexpected = [p for p in matches if p != proposal]
    # Also inspect filenames: a historical npz need not have text metadata nearby.
    filenames = subprocess.run(['rg','--files','rebuttal','local_sync',
                               '--glob','!**/fastlane_20261028_bc/**'],
                              cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()
    filename_hits = [p for p in filenames if re.search(r'(?<!\d)9900[1-5](?!\d)',p)]
    if unexpected or filename_hits:
        raise RuntimeError(f'Historical seed collision: {unexpected}, {filename_hits}')
    return dict(seeds=NEW, searched_roots=['rebuttal','local_sync'],
                strict_integer_text_matches=matches, filename_matches=filename_hits,
                planned_mentions_excluded=[proposal] if proposal in matches else [],
                historical_collisions=[], historical_scope='all visible historical experiment text and filenames under the searched roots')

def main():
    if OUT.exists(): raise RuntimeError('Inputs are immutable; use the existing frozen inputs.')
    audit = historical_seed_audit()
    OUT.mkdir(parents=True)
    original = source('S1_sparse')
    plan = read(original.parent/'plan.json')
    order = plan['candidate_order'][:20]
    assert order == [0,1,20,21,2,3,22,23,4,5,6,7,8,9,10,11,12,13,14,15]
    pool = read(original/'candidate_pool.json')
    frozen = dict(case='S1_sparse', candidate_order=order, candidate_count=20,
                  index_semantics='original candidate IDs at the first 20 positions of formal24 candidate_order',
                  methods=METHODS, development_seeds=DEV,
                  source_pool=rel(original/'candidate_pool.json'),source_pool_sha256=sha(original/'candidate_pool.json'),
                  source_plan=rel(original.parent/'plan.json'),source_plan_sha256=sha(original.parent/'plan.json'),
                  candidates={m:[dict(candidate=ci,params=pool[m][ci]) for ci in order] for m in METHODS},
                  selection='mean validation RMSE over both development seeds; both statuses ok; tie by original candidate ID; never test labels',
                  single_fit_timeout_sec=300,stage_wall_limit_sec=2700, max_parallel_single_thread_fits=7,
                  incomplete_policy='record truncation; no automatic time extension; require both development successes for eligibility')
    dump(OUT/'b1_candidate_pool_20.json', frozen)
    reused,missing,checks = [],[],[]
    dev_inputs = {}
    for seed in DEV:
        dp=ROOT/f'local_sync/fastlane_data/S1_sparse__s{seed}.npz'
        assert sha(dp)==plan['data_sha256'][dp.name]
        with np.load(dp,allow_pickle=False) as z:
            dev_inputs[seed] = dict(y_val=z['y_val'].copy(),splits={k:z[k].copy() for k in ['train','val','test']})
    for method in METHODS:
        for ci in order:
            for seed in DEV:
                rp=original/'search'/method/f'c{ci:02}'/f's{seed}'/'result.json'
                result=read(rp) if rp.exists() else {}
                base=dict(case='S1_sparse',method=method,candidate=ci,seed=seed,params=pool[method][ci],
                          data_relative_to_root=f'local_sync/fastlane_data/S1_sparse__s{seed}.npz')
                ok=result.get('status')=='ok' and math.isfinite(result.get('val_rmse',float('nan')))
                if not ok:
                    missing.append(dict(base,original_status=result.get('status','missing'),
                                        original_result_path=rel(rp) if rp.exists() else None,
                                        timeout_sec=300,evaluate_test=False))
                    continue
                assert result['requested_params']==base['params']
                assert result['worker_sha256']==plan['worker_sha256']
                assert result['seed']==seed and result['method']==method
                assert result.get('evaluate_test') is False and result.get('test_labels_used') is False
                assert 'test' not in result.get('metrics',{}) and 'test' not in result.get('target_hashes',{})
                pp=rp.parent/'predictions.npz'
                with np.load(pp,allow_pickle=False) as z:
                    assert 'test' not in z.files
                    vk='val' if 'val' in z.files else 'pred_val'
                    pred=z[vk]
                    rmse=float(np.sqrt(np.mean((pred-dev_inputs[seed]['y_val'])**2)))
                assert abs(rmse-result['val_rmse'])<1e-8,(rp,rmse,result['val_rmse'])
                copied=OUT/'reused_results'/method/f'c{ci:02}'/f's{seed}.json'
                copied.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(rp,copied)
                reused.append(dict(base,status='ok',val_rmse=result['val_rmse'],selection_value=result['val_rmse'],
                                   result_path=rel(rp),result_sha256=sha(rp),frozen_result=str(copied.relative_to(OUT)),
                                   prediction_path=rel(pp),prediction_sha256=sha(pp)))
                checks.append(dict(method=method,candidate=ci,seed=seed,val_rmse_recomputed=rmse))
    assert len(missing)<=30
    assert all(j['method']=='LinXGBoost' for j in missing)
    assert len(reused)+len(missing)==6*20*2
    dump(OUT/'b1_reused_results.json',dict(records=reused,count=len(reused),selection_is_pending=True))
    dump(OUT/'b1_missing_jobs.json',dict(jobs=missing,count=len(missing),max_jobs=30,evaluate_test=False))
    c_choices={}
    for case in ['S2_sparse','S3_sparse']:
        cp=source(case)/'choice.json';chosen=read(cp)
        cp_pool=read(source(case)/'candidate_pool.json')
        for method in METHODS:
            item=chosen['choices'][method]
            assert item['params']==cp_pool[method][item['candidate']]
        c_choices[case]=dict(source_choice=rel(cp),source_choice_sha256=sha(cp),
                            source_candidate_pool=rel(source(case)/'candidate_pool.json'),
                            source_candidate_pool_sha256=sha(source(case)/'candidate_pool.json'),
                            choices={m:{k:v for k,v in chosen['choices'][m].items() if k in ['candidate','params','mean_development_primary','selection_metric']} for m in METHODS})
    dump(OUT/'c_frozen_choices.json',dict(stage='C',execute_after='B_complete',cases=c_choices,
        confirmation_seeds=NEW,methods=METHODS,leafonly='same CrossGB params; conditional control; no HPO',
        selection='original formal24 frozen choices only; no new selection on C data'))
    generator=ROOT/'rebuttal/fastlane_20261027/prepare_data.py'
    spec=importlib.util.spec_from_file_location('frozen_data_generator',generator)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    data_entries=[];data_checks=[]
    with tempfile.TemporaryDirectory(prefix='bc-inputs-') as temporary:
        temp=Path(temporary)
        generated=mod.synthetic(temp,NEW,.3,ROOT)
        assert len(generated)==45
        for entry in generated:
            if '_sparse__' not in entry['dataset']:continue
            name=entry['dataset'];meta=read(temp/(name+'.json'))
            assert meta['seed'] in NEW and meta['phase']=='confirmation'
            assert meta['control']=='sparse' and meta['noise_sd']==.3
            assert not meta['oracle_support_in_X'] and not meta['groups_in_X']
            with np.load(temp/(name+'.npz'),allow_pickle=False) as z:
                x=z['X']; n,d=x.shape; family=meta['family']
                assert (n,d)==({'S1':(240,32),'S2':(480,32),'S3':(480,16)}[family])
                # Reproduce X and noise from the exact substream, without reading test targets.
                rng=np.random.default_rng(np.random.SeedSequence([meta['seed'],int(family[1:]),100]))
                raw_x=rng.normal(size=(n,d));eps=rng.normal(scale=.3,size=n)
                assert np.array_equal(x,raw_x.astype(np.float32))
                assert np.array_equal(z['noise'],eps)
                parts=[z[k] for k in ['train','val','test']]
                assert np.array_equal(np.sort(np.concatenate(parts)),np.arange(n))
                assert sum(len(set(v.tolist())) for v in parts)==len(set(np.concatenate(parts).tolist()))==n
                assert all(np.array_equal(a,b) for a,b in zip(parts,mod.random_split(n,meta['seed'])))
                assert z.files==['X','y','train','val','test','task_type','oracle_region','oracle_signal','noise','y_train','y_val','y_test']
                for key,expected in meta['array_sha256'].items():
                    # Do not inspect confirmation test target values; generation supplied their frozen hashes.
                    if key in ['y','y_test']:continue
                    assert mod.ahash(z[key])==expected
            target=OUT/'data';target.mkdir(exist_ok=True)
            for ext in ['.npz','.json']:shutil.copyfile(temp/(name+ext),target/(name+ext))
            data_entries.append(dict(entry, npz='data/'+name+'.npz',metadata='data/'+name+'.json',
                                     metadata_sha256=sha(target/(name+'.json'))))
            data_checks.append(dict(dataset=name,shape=[n,d],split_sizes=meta['split_sizes'],
                                    X_matches_original_substream=True,noise_matches_original_substream=True,
                                    splits_disjoint_and_exhaustive=True,oracle_not_in_X=True,
                                    confirmation_test_targets_not_inspected=True))
    assert len(data_entries)==15
    validation=dict(status='passed',no_model_import_or_fit=True,no_network=True,
                    old_inputs_unmodified=True,historical_seed_audit=audit,
                    reused_result_validation=dict(count=len(checks),all_validation_rmse_recomputed=True,
                        maximum_abs_error=0 if not checks else max(abs(c['val_rmse_recomputed']-next(r['val_rmse'] for r in reused if all(r[k]==c[k] for k in ['method','candidate','seed']))) for c in checks),
                        no_test_metrics_or_target_hashes_in_reused_search=True),
                    data=data_checks,B_recipe_status='pending B1 completion; deliberately not selected',
                    C_recipe_status='original formal24 choices frozen')
    dump(OUT/'validation.json',validation)
    manifest=dict(protocol_version='fastlane-bc-20261028-inputs-v1',
                  root_relative_to_inputs='../../..',development_seeds=DEV,confirmation_seeds=NEW,
                  stage_order=['B1','freeze_all_B_choices','B2','complete_B','C'],
                  b1=dict(case='S1_sparse',pool='b1_candidate_pool_20.json',candidate_order=order,
                          reused_results='b1_reused_results.json',reused_fit_count=len(reused),
                          missing_jobs='b1_missing_jobs.json',missing_fit_count=len(missing),
                          data='existing immutable local_sync/fastlane_data/S1_sparse__s97001/97002.npz',
                          selection_status='pending',requires_test_labels=False),
                  b2=dict(case='S1_sparse',recipe_status='must be frozen after B1 before any B2 evaluation',
                          expected_final_fits=30,expected_leafonly_fits=5),
                  c=dict(choices='c_frozen_choices.json',execute_after='B_complete',expected_final_fits=60,expected_leafonly_fits=10),
                  generator=dict(path=rel(generator),sha256=sha(generator),function='synthetic',
                                 noise_sd=.3,temporary_controls_generated=45,retained_sparse_count=15),
                  datasets=data_entries,validation='validation.json',
                  safety=dict(no_hpo_on_confirmation=True,test_gate='freeze all B choices before B2 test; C original frozen choices; no test-based recipe changes',
                              oracle_arrays_diagnostic_only=True,no_training_executed=True),
                  prepared_script_sha256=sha(Path(__file__)))
    dump(OUT/'manifest.json',manifest)
    hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    dump(OUT/'sha256.json',hashes)
    print(json.dumps(dict(manifest=str(OUT/'manifest.json'),reused=len(reused),missing=len(missing),
                          npz=len(data_entries),candidate_order=order,validation='passed')))

if __name__=='__main__':main()
