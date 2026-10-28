#!/usr/bin/env python3
"""Aggregate existing fastlane files; never fit, select, or inspect predictions."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

CONFIRMATION = tuple(range(98001, 98006))
SPLITS = ('train', 'val', 'test')
FOLD_FIELDS = ['suite','case','family','source_kind','stage','configuration','method','seed','split','metric',
               'value','value_standardized','metric_units','y_scale','status','error','test_labels_used',
               'n_train','n_val','n_test','fit_wall_sec','fit_cpu_sec','predict_wall_sec','predict_cpu_sec',
               'total_wall_sec','test_index_hash','shared_real_test','record_sha256','data_protocol','source_file']
SUMMARY_FIELDS = ['suite','case','family','source_kind','stage','configuration','method','split','metric','metric_units',
                  'n_expected','n_observed','n_success','n_failed','n_pending','n_metric_missing','n_absent',
                  'coverage','complete','mean','std','min','max','fit_wall_mean_sec','fit_cpu_mean_sec',
                  'predict_wall_mean_sec','test_coverage','shared_real_test','repeats_interpretation','seeds_success']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def write_csv(path, rows, fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def locate(path,record,root):
    """Use the documented directory contract, then recorded data basename."""
    parts=path.relative_to(root).parts
    stage='capacity' if path.name=='capacity.json' else ('final' if 'final' in parts else 'development')
    marker='capacity' if stage=='capacity' else ('final' if stage=='final' else ('search' if 'search' in parts else None))
    if marker in parts:
        pos=parts.index(marker);case=parts[pos-1] if pos else ''
        ancestors=parts[:max(pos-1,0)]
    else:
        choices=[v for v in path.parents if (v/'choice.json').exists()]
        case=choices[0].name if choices else ''
        ancestors=parts[:-3]
    stem=Path(record.get('data','')).stem
    if not case and '__s' in stem:case=stem.rsplit('__s',1)[0]
    if not case:case=path.parent.parent.name
    seed=record.get('seed')
    if seed is None:
        matches=[int(m.group(1)) for v in reversed(parts) if (m:=re.fullmatch(r's(\d+)',v))]
        seed=matches[0] if matches else None
    suite=record.get('suite') or next((v for v in reversed(root.parts+ancestors) if re.search(r'(quick|formal)',v,re.I)), 'unspecified')
    return str(suite),case,stage,seed,stem

def read_sidecar(data,case,seed,stem,audit):
    candidates=[data/(stem+'.json')] if stem else []
    candidates.append(data/f'{case}__s{seed}.json')
    for path in candidates:
        if path.exists():
            try:return json.loads(path.read_text()),path
            except (ValueError,OSError) as error:
                audit['invalid_metadata'].append({'path':str(path),'error':str(error)})
    audit['missing_metadata'].append({'case':case,'seed':seed,'data_stem':stem})
    return {},None

def normalize(path,record,root,data,audit):
    suite,case,stage,seed,stem=locate(path,record,root)
    meta,metadata_path=read_sidecar(data,case,seed,stem,audit)
    task=record.get('task_type',meta.get('task_type','regression'))
    synthetic=bool(meta.get('family')) or bool(re.match(r'^S[123]_',case))
    source=meta.get('source',{})
    provenance=source.get('provenance') if isinstance(source,dict) else meta.get('provenance')
    source_kind='synthetic' if synthetic else ('simulation' if provenance=='simulation' else 'real')
    family=meta.get('family',case)
    scale=meta.get('preprocessing',{}).get('y_scale',1.)
    if not finite(scale) or scale<=0:
        audit['invalid_metadata'].append({'path':str(metadata_path),'error':'invalid y_scale'})
        scale=None
    has_scale=synthetic or bool(metadata_path and 'y_scale' in meta.get('preprocessing',{}))
    if task=='regression' and not has_scale:
        # Do not silently label standardized values as source units.
        scale=None
    digest=sha(path.read_bytes())
    params=record.get('requested_params',record.get('effective_params',{}))
    candidate=next((v for v in path.parts if re.fullmatch(r'c\d+',v)),None)
    configuration='selected' if stage in ('final','capacity') else candidate or sha(json.dumps(params,sort_keys=True,default=str).encode())[:12]
    base=dict(suite=suite,case=case,family=family,source_kind=source_kind,stage=stage,configuration=configuration,
              seed=seed,status=record.get('status','unknown'),error=record.get('error',''),
              test_labels_used=record.get('test_labels_used'),y_scale=scale,
              shared_real_test=not synthetic,test_index_hash=meta.get('array_sha256',{}).get('test'),
              record_sha256=digest,data_protocol=meta.get('protocol_version'),source_file=str(path))
    counts=record.get('split_counts',record.get('split_sizes',meta.get('split_sizes',{})))
    base.update({'n_'+s:counts.get(s) for s in SPLITS})
    if stage=='development' and (record.get('test_labels_used') or record.get('metrics',{}).get('test')):
        audit['selection_test_exposure'].append({'path':str(path),'case':case,'seed':seed})
    rows=[]
    if stage=='capacity':
        valid_status=base['status']=='completed_fixed_route_probe'
        for method,entry in record.get('metrics',{}).items():
            times=entry.get('fit_timing',{})
            for split,value in entry.get('rmse',{}).items():
                row=dict(base,method=method,split=split,metric='rmse',value_standardized=value,
                         value=value*scale if finite(value) and scale is not None else None,
                         metric_units='source_target' if scale is not None else 'standardized_unknown_scale',
                         fit_wall_sec=times.get('wall_seconds'),fit_cpu_sec=times.get('cpu_seconds'),
                         predict_wall_sec=entry.get('prediction_timing',{}).get(split,{}).get('wall_seconds'),
                         predict_cpu_sec=entry.get('prediction_timing',{}).get(split,{}).get('cpu_seconds'))
                row['status']='ok' if valid_status else base['status'];rows.append(row)
    else:
        parts=path.parts
        path_method=parts[parts.index('search')+1] if 'search' in parts else path.parent.parent.name
        method=record.get('method',path_method)
        timing=record.get('timings',{})
        splits=SPLITS if stage=='final' or record.get('evaluate_test') else ('train','val')
        expected_metrics=['rmse'] if task=='regression' else (['auc','logloss','accuracy'] if task=='binary' else ['logloss','accuracy'])
        for split in splits:
            measured=record.get('metrics',{}).get(split,{})
            for metric in expected_metrics:
                value=measured.get(metric,record.get(split+'_'+metric))
                units='dimensionless'
                converted=value if finite(value) else None
                if metric=='rmse':
                    converted=value*scale if finite(value) and scale is not None else None
                    units='source_target' if scale is not None else 'standardized_unknown_scale'
                rows.append(dict(base,method=method,split=split,metric=metric,
                    value=converted,value_standardized=value if finite(value) else None,metric_units=units,
                    fit_wall_sec=timing.get('fit_wall_sec'),fit_cpu_sec=timing.get('fit_cpu_sec'),
                    predict_wall_sec=timing.get('predict_'+split+'_wall_sec'),predict_cpu_sec=timing.get('predict_'+split+'_cpu_sec'),
                    total_wall_sec=record.get('total_wall_sec')))
    return rows

def summary_rows(rows,expected_seeds):
    groups=defaultdict(list)
    keys=('suite','case','family','source_kind','stage','configuration','method','split','metric','metric_units')
    for row in rows:groups[tuple(row.get(k) for k in keys)].append(row)
    result=[]
    for key,group in sorted(groups.items()):
        seeds=set(expected_seeds) if key[4] in ('final','capacity') else {r['seed'] for r in group}
        observed={r['seed'] for r in group if r['seed'] in seeds}
        success=[r for r in group if r['seed'] in seeds and r['status']=='ok' and finite(r['value'])]
        values=[r['value'] for r in success]
        failures={r['seed'] for r in group if r['seed'] in seeds and r['status'] not in ('ok','running')}
        pending={r['seed'] for r in group if r['seed'] in seeds and r['status']=='running'}
        missing={r['seed'] for r in group if r['seed'] in seeds and r['status']=='ok' and not finite(r['value'])}
        row=dict(zip(keys,key))
        n=len(values);nexp=len(seeds)
        row.update(n_expected=nexp,n_observed=len(observed),n_success=n,n_failed=len(failures),n_pending=len(pending),
                   n_metric_missing=len(missing),n_absent=len(seeds-observed),coverage=f'{n}/{nexp}',
                   complete=n==nexp,mean=statistics.mean(values) if values else None,
                   std=statistics.stdev(values) if n>1 else None,min=min(values) if values else None,max=max(values) if values else None,
                   test_coverage=f'{n}/{nexp}' if key[7]=='test' else None,shared_real_test=group[0]['shared_real_test'],
                   repeats_interpretation='共享固定测试集的训练/验证重复，不能视为独立数据集' if group[0]['shared_real_test'] else '各生成种子对应独立合成数据',
                   seeds_success=','.join(str(r['seed']) for r in sorted(success,key=lambda r:r['seed'])))
        for field,outfield in [('fit_wall_sec','fit_wall_mean_sec'),('fit_cpu_sec','fit_cpu_mean_sec'),('predict_wall_sec','predict_wall_mean_sec')]:
            vals=[r[field] for r in success if finite(r.get(field))]
            row[outfield]=statistics.mean(vals) if vals else None
        result.append(row)
    return result

def findings(summaries,audit):
    lines=['# Fastlane 结果汇总','',
           '验证集与测试集分别汇总；RMSE 已按各训练重复的 y_scale 换回原始目标单位。缺少缩放元数据时不输出伪装为原单位的 RMSE。',
           '真实数据及已有模拟数据的重复共享固定测试集，不能作为独立新数据集；合成确认种子生成独立数据。kin8nm 来源为模拟系统，单列 simulation，不计入真实优势数据数量。容量探针固定树路由，仿射叶子近似不代表完整 k>1 算法。','']
    final=[r for r in summaries if r['stage']=='final' and r['split']=='test']
    if not final:lines.append('尚无可汇总的最终测试结果。')
    else:
        lines+=['| 套件 / 数据 | 方法 | 指标 | 成功覆盖 | 均值 ± 样本标准差 | 状态 |','|---|---|---|---|---|---|']
        for r in final:
            mean='—' if r['mean'] is None else f"{r['mean']:.6g}"
            sd='—' if r['std'] is None else f"{r['std']:.3g}"
            state='完整' if r['complete'] else f"不完整：失败{r['n_failed']}，缺失{r['n_absent']}，待完成{r['n_pending']}，指标缺失{r['n_metric_missing']}"
            lines.append(f"| {r['suite']} / {r['case']} | {r['method']} | {r['metric']} | {r['coverage']} | {mean} ± {sd} | {state} |")
    lines+=['','上述不完整行仅描述已成功的重复，不冒充完整计划结果；不得据不完整覆盖给出稳定性或总体胜出结论。',
            f"审计：读取结果文件 {audit['result_files']}，容量文件 {audit['capacity_files']}；镜像重复 {len(audit['mirror_duplicates'])}，冲突组 {len(audit['conflicting_records'])}，缺失元数据 {len(audit['missing_metadata'])}，开发阶段测试暴露 {len(audit['selection_test_exposure'])}。",
            '训练耗时为原始单次拟合计时；容量探针拟合耗时不含既有树结构训练或共享路由计算，不能与完整模型训练直接比较。']
    if audit.get('no_complete_candidate'):
        lines+=['','未获得完整开发候选、未进入最终确认的方法：'+ '；'.join(f"{r['case']} / {r['method']}" for r in audit['no_complete_candidate'])+'。']
    return '\n'.join(lines)+'\n'

def aggregate(roots,data,output,expected_seeds=CONFIRMATION):
    output.mkdir(parents=True,exist_ok=True)
    audit=dict(result_files=0,capacity_files=0,invalid_json=[],invalid_metadata=[],missing_metadata=[],
               mirror_duplicates=[],conflicting_records=[],selection_test_exposure=[],choices=[],no_complete_candidate=[],record_status_counts={},
               expected_confirmation_seeds=list(expected_seeds),policy='Conflicting same logical fits are excluded, never outcome-selected; partial means are labeled by coverage.')
    candidates=defaultdict(list)
    seen_paths=set()
    for root in roots:
        if not root.exists():
            audit['invalid_json'].append({'path':str(root),'error':'results root absent'});continue
        for path in sorted(root.rglob('choice.json')):
            try:
                choice=json.loads(path.read_text())
                audit['choices'].append({'path':str(path),'choice':choice})
                for method,selection in choice.get('choices',{}).items():
                    if selection.get('status')=='no_complete_candidate':
                        audit['no_complete_candidate'].append({'case':choice.get('case',path.parent.name),'method':method,'source':str(path)})
            except (ValueError,OSError) as error:audit['invalid_json'].append({'path':str(path),'error':str(error)})
        for name in ('result.json','capacity.json'):
            for path in sorted(root.rglob(name)):
                if path.resolve() in seen_paths:continue
                seen_paths.add(path.resolve())
                audit['result_files' if name=='result.json' else 'capacity_files']+=1
                try:
                    record=json.loads(path.read_text())
                    status=record.get('status','unknown')
                    audit['record_status_counts'][status]=audit['record_status_counts'].get(status,0)+1
                    rows=normalize(path,record,root,data,audit)
                except (ValueError,OSError,TypeError,KeyError) as error:
                    audit['invalid_json'].append({'path':str(path),'error':str(error)});continue
                for row in rows:
                    identity=tuple(row[k] for k in ('suite','case','stage','configuration','method','seed','split','metric'))
                    candidates[identity].append(row)
    folds=[]
    for key,items in sorted(candidates.items(),key=lambda pair:str(pair[0])):
        hashes={r['record_sha256'] for r in items}
        if len(hashes)>1:
            audit['conflicting_records'].append({'identity':list(key),'paths':[r['source_file'] for r in items]})
            continue
        folds.append(items[0])
        if len(items)>1:audit['mirror_duplicates'].append({'identity':list(key),'paths':[r['source_file'] for r in items]})
    summaries=summary_rows(folds,expected_seeds)
    # Fixed-test consistency uses data-sidecar index hashes, not labels or scores.
    tests=defaultdict(set)
    for row in folds:
        if row['source_kind']=='real' and row.get('test_index_hash'):
            tests[(row['suite'],row['case'])].add(row['test_index_hash'])
    audit['real_test_hash_counts']=[{'suite':k[0],'case':k[1],'distinct_test_index_hashes':len(v),'fixed':len(v)==1} for k,v in sorted(tests.items())]
    audit['rows_emitted']=len(folds);audit['summary_rows']=len(summaries)
    audit['complete_final_test_groups']=sum(r['complete'] for r in summaries if r['stage']=='final' and r['split']=='test')
    write_csv(output/'folds.csv',folds,FOLD_FIELDS)
    write_csv(output/'summary.csv',summaries,SUMMARY_FIELDS)
    write_json(output/'audit.json',audit)
    (output/'findings_zh.md').write_text(findings(summaries,audit))
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,nargs='+',required=True,help='One or more recursively searched host/download roots')
    p.add_argument('--data',type=Path,required=True,help='Directory containing <case>__s<seed>.json sidecars')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-seeds',type=int,nargs='+',default=list(CONFIRMATION))
    args=p.parse_args()
    audit=aggregate([r.resolve() for r in args.results],args.data.resolve(),args.output.resolve(),args.expected_seeds)
    print(json.dumps({k:audit[k] for k in ['result_files','capacity_files','rows_emitted','summary_rows','complete_final_test_groups']}))

if __name__=='__main__':main()
