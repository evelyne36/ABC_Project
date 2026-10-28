"""Recompute the newly synced evidence. Writes only this analysis directory."""
from pathlib import Path
import collections, csv, hashlib, json, math, statistics, subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
LIVE = ROOT.parent / 'local_sync/remote_audit_20261027'
EVIDENCE = ROOT / '过程汇报_checkpoint_1027/证据_远端'

def dumpcsv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def mean(xs): return statistics.mean(xs) if xs else None
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checks={}
    base=EVIDENCE/'N3_index'
    reports=json.loads((base/'all_budget_results.json').read_text())
    assert len(reports)==30
    assert len({(x['dataset'],x['method'],x['outer_fold']) for x in reports})==30
    plan=json.loads(next((LIVE/'TabularData_Processor').rglob('N3_equal_search/plan.json')).read_text())
    rows=[]; mismatches=[]
    for r in reports:
        name=f"{r['dataset']}__outer{r['outer_fold']}__{r['method']}"
        individual=json.loads((base/name/'budget_results.json').read_text())
        assert r==individual
        trace=json.loads((base/name/'search_trace.json').read_text())
        assert [x['candidate_index'] for x in trace]==plan['candidate_order'][:len(trace)]
        for cp in r['checkpoints']:
            eligible=[x for x in trace if x['status']=='success' and x['completed_search_wall_sec']<=cp['search_budget_sec']]
            best=min(eligible,key=lambda x:(x['validation_rmse'],x['order'])) if eligible else None
            assert len(eligible)==cp['completed_candidates']
            assert cp['selected_candidate_index']==(best['candidate_index'] if best else None)
            if best: assert abs(cp['validation_rmse']-best['validation_rmse'])<1e-10
            fr=cp.get('final_refit') or {}
            rows.append(dict(dataset=r['dataset'],method=r['method'],fold=r['outer_fold'],budget=cp['search_budget_sec'],
                completed_candidates=cp['completed_candidates'],selected_candidate=cp['selected_candidate_index'],
                validation_rmse=cp['validation_rmse'],status=fr.get('status','no_completed_candidate'),
                test_rmse=fr.get('test_rmse'),final_fit_wall=fr.get('fit_wall_sec'),final_fit_cpu=fr.get('fit_cpu_sec'),
                total_cap_plus_refit_wall=cp.get('search_cap_plus_selected_refit_wall_sec')))
        remote=LIVE/'TabularData_Processor/rebuttal_formal_20261026/results/N3_equal_search'/name/'budget_results.json'
        if remote.exists() and digest(remote)!=digest(base/name/'budget_results.json'): mismatches.append(name)
    assert not mismatches
    dumpcsv('N3_all_90_checkpoints.csv',rows)
    groups=collections.defaultdict(list)
    for r in rows: groups[(r['dataset'],r['budget'],r['method'])].append(r)
    agg=[]
    for (dataset,budget,method),rs in sorted(groups.items()):
        valid=[r for r in rs if r['status']=='success' and r['test_rmse'] is not None]
        agg.append(dict(dataset=dataset,budget=budget,method=method,folds=len(rs),successful_folds=len(valid),
            mean_test_rmse=mean([r['test_rmse'] for r in valid]),
            mean_completed_candidates=mean([r['completed_candidates'] for r in rs]),
            mean_final_fit_wall=mean([r['final_fit_wall'] for r in valid]),
            mean_final_fit_cpu=mean([r['final_fit_cpu'] for r in valid]),
            mean_total_cap_plus_refit_wall=mean([r['total_cap_plus_refit_wall'] for r in valid])))
    dumpcsv('N3_summary.csv',agg)
    checks['N3']={'groups':len(reports),'checkpoints':len(rows),'trace_selection_reconstruction':'all_passed',
        'live_budget_file_hash_mismatches':mismatches,'candidate_order':'one frozen order, seed42; no order robustness established',
        'cpu_900_valid_finite_negative_results':[r['test_rmse'] for r in rows if r['dataset']=='cpu_activity' and r['method']=='current_metagb' and r['budget']==900]}

    # Use the live full-precision summaries, rather than rounded transcription.
    lgb=[]; seen=set()
    for p in LIVE.glob('*/rebuttal_formal_20261026/results/N1/lgb_*/*/outer*/summary.json'):
        r=json.loads(p.read_text()); key=(r['dataset'],r['method'],r.get('outer_fold'))
        if key in seen: continue
        seen.add(key)
        lgb.append(dict(dataset=key[0],method=key[1],fold=key[2],test_rmse=r['test_rmse'],
            inner_rmse=r['selected_mean_inner_rmse'],selected_candidate=r['selected_candidate_index'],
            failed_candidates=r['failed_candidates'],standardize=r['standardize'],
            candidate_count=r['candidate_count'],source=str(p.relative_to(ROOT.parent))))
    assert len(lgb)==50
    dumpcsv('LGB_50_folds.csv',sorted(lgb,key=lambda x:(x['dataset'],x['method'],x['fold'])))
    gg=collections.defaultdict(list)
    for r in lgb: gg[(r['dataset'],r['method'])].append(r)
    la=[]
    for (dataset,method),rs in sorted(gg.items()):
        assert len(rs)==5
        la.append(dict(dataset=dataset,method=method,folds=5,mean_rmse=mean([r['test_rmse'] for r in rs]),
            median_rmse=statistics.median(r['test_rmse'] for r in rs),maximum_rmse=max(r['test_rmse'] for r in rs)))
    dumpcsv('LGB_full_fivefold_summary.csv',la)
    checks['LGB']={'folds':len(lgb),'all_summaries_standardize_false':all(not r['standardize'] for r in lgb),
        'cpu_linear_outer4':next(r for r in lgb if r['dataset']=='cpu_activity' and r['method']=='lgb_linear' and r['fold']==4)}

    # Exact pooled OOF loss reconstructed from fold metrics and test counts.
    oldpath=ROOT/'aaai_FullCodeProject-check_codereview0701/datacollection0702/0716dataAddition/metagb_ablation_syn/syn_formal_formal16_v1/fold_results.csv'
    old=list(csv.DictReader(oldpath.open()))
    taskmeta={r['task_id']:r for r in old if r['dimension']=='10'}
    variants={'xgb':'XGB','bias_p00_sa0':'Bias','split_p10_sa0':'SplitOnly','full_p10_sa0':'Full'}
    vals=[]
    for r in old:
        if r['dimension']=='10' and r['variant_id'] in variants:
            vals.append(dict(scenario=r['scenario'],seed=int(r['data_seed']),fold=int(r['outer_fold']),
                method=variants[r['variant_id']],metric=r['primary_metric'],n=int(r['n_test']),value=float(r['test_primary_metric'])))
    for p in (EVIDENCE/'N2_leafonly').glob('*/result.json'):
        r=json.loads(p.read_text()); assert r['status']=='success'; meta=taskmeta[r['task_id']]
        vals.append(dict(scenario=meta['scenario'],seed=int(meta['data_seed']),fold=int(meta['outer_fold']),
            method='LeafOnly',metric=r['metric_name'],n=int(meta['n_test']),value=r['metric']))
    sg=collections.defaultdict(list)
    for r in vals:sg[(r['scenario'],r['seed'],r['method'],r['metric'])].append(r)
    seeds=[]
    for (scenario,seed,method,metric),rs in sorted(sg.items()):
        assert len(rs)==5
        loss=sum(r['n']*(r['value']**2 if metric=='rmse' else r['value']) for r in rs)/sum(r['n'] for r in rs)
        seeds.append(dict(scenario=scenario,seed=seed,method=method,metric=metric,pooled_oof_metric=math.sqrt(loss) if metric=='rmse' else loss))
    dumpcsv('N2_seed_level_OOF.csv',seeds)
    ng=collections.defaultdict(list)
    for r in seeds:ng[(r['scenario'],r['method'],r['metric'])].append(r['pooled_oof_metric'])
    nr=[dict(scenario=k[0],method=k[1],metric=k[2],independent_seeds=len(v),mean_pooled_oof=mean(v),sd=statistics.stdev(v)) for k,v in sorted(ng.items())]
    dumpcsv('N2_OOF_summary.csv',nr)
    checks['N2']={'seed_method_cells':len(seeds),'groups':len(nr),'independent_seed_count':3,
        'pooling':'sqrt(sum(n_test*RMSE^2)/sum(n_test)); logloss weighted by n_test',
        'max_Bias_minus_XGB_pooled_oof':max(abs(a['pooled_oof_metric']-b['pooled_oof_metric']) for a in seeds for b in seeds if a['method']=='Bias' and b['method']=='XGB' and (a['scenario'],a['seed'])==(b['scenario'],b['seed']))}

    # Restore only byte-identical missing historical TS inputs.
    manifest=json.loads((ROOT/'rebuttal_server_config_20261025/reports/input_transfer_manifest.json').read_text())['files']
    restored=[]
    for row in manifest:
        if Path(row['path']).name not in ['ts_selected_models.json','ts_submitted_split_manifest.json']:continue
        src=LIVE/'TabularData_Processor'/row['path'];dst=ROOT/row['path']
        assert digest(src)==row['sha256']
        if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
        assert digest(dst)==row['sha256'];restored.append({'path':row['path'],'sha256':row['sha256']})
    checks['restored_historical_TS_files']=restored
    structural=ROOT.parent/'local_sync/remote_structure_20261027/cpu4_pool.zjuici.com.json'
    if structural.exists():
        source=json.loads(structural.read_text())['data']['leaf_files']
        assert len(source)==20
        groups=collections.defaultdict(list)
        for r in source:groups[r['path'].split('/')[-2].rsplit('__outer',1)[0]].append(r)
        structural_rows=[]
        for key,rs in sorted(groups.items()):
            assert len(rs)==5
            counts=collections.Counter()
            for r in rs:counts.update({int(k):v for k,v in r['support_counts'].items()})
            total=sum(counts.values());targets=[(total-1)//2,total//2];position=0;middle=[]
            for size,count in sorted(counts.items()):
                for target in targets:
                    if position<=target<position+count:middle.append(size)
                position+=count
            predictive=sum(r['predictive'] for r in rs);outside=sum(r['outside_predictive'] for r in rs)
            structural_rows.append(dict(group=key,folds=5,leaves=total,predictive_leaves=predictive,
                outside_predictive_leaves=outside,outside_fraction_among_predictive=outside/predictive if predictive else None,
                median_training_rows_per_leaf=mean(middle)))
        dumpcsv('N4_verified_structure.csv',structural_rows)
        checks['N4']={'remote_leaf_files':20,'aggregation':'pooled over leaves; differs from unweighted mean of per-fold fractions','rows':structural_rows}
    checks['git_head']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    checks['raw_source_hashes']={str(p.relative_to(ROOT)):digest(p) for p in [base/'all_budget_results.json',oldpath]}
    (OUT/'audit.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(checks,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
