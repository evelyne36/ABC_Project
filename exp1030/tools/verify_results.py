#!/usr/bin/env python3
# coding: utf-8
"""Verify archive hashes and independently recompute every successful final RMSE.

Run from any working directory: python exp1030/tools/verify_results.py
Requires NumPy; never imports estimator code or unpickles models.
"""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['CrossGB', 'PL', 'LinXGBoost', 'PILOT', 'TAO']


def read_bytes(tar, name):
    if isinstance(tar, dict):
        return tar[name]
    f = tar.extractfile(name)
    if f is None:
        raise ValueError('Missing archive member: ' + name)
    return f.read()


def read_json(tar, name):
    return json.loads(read_bytes(tar, name))


def inspect_archive(path):
    trials = []
    checks = []
    statuses = collections.Counter()
    with tarfile.open(path, 'r:gz') as tar:
        names = tar.getnames()
        assert len(names) == len(set(names)), 'Duplicate archive paths'
        for name in names:
            assert not name.startswith('/') and '..' not in Path(name).parts, 'Unsafe archive path'
        payloads = {member.name: tar.extractfile(member).read() for member in tar.getmembers()
                    if member.isfile()}
        tar = payloads
        manifest = read_json(tar, 'EXPORT_MANIFEST.json')
        assert set(names) == {x['path'] for x in manifest['files']} | {'EXPORT_MANIFEST.json'}
        for item in manifest['files']:
            data = read_bytes(tar, item['path'])
            assert len(data) == item['bytes'], item['path']
            assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
        for name in names:
            if not name.startswith('results/') or not name.endswith('/result.json'):
                continue
            _, case, stage, seed, method, index, _ = name.split('/')
            record = read_json(tar, name)
            status = record.get('status', 'unknown')
            statuses[status] += 1
            meta_name = f'inputs/{case}/{stage}_{seed}.json'
            meta = read_json(tar, meta_name)
            scale = meta['y_scale']
            params_name = name.rsplit('/', 1)[0] + '/params.json'
            params = read_json(tar, params_name)
            def raw(metric):
                value = record.get(metric + '_rmse')
                if value is None:
                    value = record.get('metrics', {}).get(metric, {}).get('rmse')
                return value
            def original(metric):
                value = raw(metric)
                return value * scale if value is not None else None
            row = dict(archive=path.name, case=case, stage=stage, seed=int(seed),
                       method=method, candidate_index=int(index), status=status,
                       train_rmse_normalized=raw('train'),
                       val_rmse_normalized=raw('val'), test_rmse_normalized=raw('test'),
                       train_rmse_original=original('train'), val_rmse_original=original('val'),
                       test_rmse_original=original('test'), y_scale=scale,
                       fit_cpu_sec=record.get('timings', {}).get('fit_cpu_sec'),
                       fit_wall_sec=record.get('timings', {}).get('fit_wall_sec'),
                       predict_test_cpu_sec=record.get('timings', {}).get('predict_test_cpu_sec'),
                       requested_params_json=json.dumps(params, sort_keys=True),
                       result_path_in_archive=name,
                       result_sha256=hashlib.sha256(read_bytes(tar, name)).hexdigest(),
                       input_npz_sha256=meta.get('npz_sha256'),
                       worker_sha256=record.get('worker_sha256'),
                       error_type=record.get('error_type'), error=record.get('error'))
            trials.append(row)
            if stage != 'confirm':
                assert not record.get('test_labels_used') and record.get('test_rmse') is None
            if status != 'ok' or stage != 'confirm':
                continue
            frozen = read_json(tar, f'frozen/{case}.json')
            assert params == record['requested_params'] == frozen['selected'][method]
            prediction_name = name.rsplit('/', 1)[0] + '/predictions.npz'
            target_name = f'evaluation_targets/{case}/confirm_{seed}.npz'
            with np.load(io.BytesIO(read_bytes(tar, prediction_name)), allow_pickle=False) as pred:
                prediction = pred['test']
                indices = pred['test_indices']
            with np.load(io.BytesIO(read_bytes(tar, target_name)), allow_pickle=False) as truth:
                target = truth['y_test']
                assert np.array_equal(indices, truth['test_indices'])
            assert prediction.shape == target.shape and np.isfinite(prediction).all()
            value = float(np.sqrt(np.mean((prediction - target) ** 2)))
            error = abs(value - record['test_rmse'])
            assert np.isclose(value, record['test_rmse'], rtol=1e-7, atol=1e-9), name
            checks.append(dict(archive=path.name, case=case, seed=int(seed), method=method,
                               rmse_original=value*scale, rmse_normalized=value,
                               recorded_rmse_normalized=record['test_rmse'], error_abs=error,
                               n_test=len(target), input_npz_sha256=meta['npz_sha256']))
        assert dict(statuses) == manifest['counts']
    return trials, checks, dict(statuses)


def verify():
    checksum_file = ROOT / 'SHA256SUMS'
    if checksum_file.exists():
        for line in checksum_file.read_text().splitlines():
            digest, name = line.split('  ', 1)
            file = ROOT / name
            assert file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() == digest, name
    trials, checks, per_archive = [], [], {}
    for path in sorted((ROOT / 'raw_archives').glob('pool*.tar.gz')):
        t, c, s = inspect_archive(path)
        trials.extend(t)
        checks.extend(c)
        per_archive[path.name] = s
    assert len(per_archive) == 6
    statuses = dict(collections.Counter(r['status'] for r in trials))
    expected = json.loads((ROOT / 'provenance/collection_manifest.json').read_text())
    assert statuses == expected['trial_status_counts']
    assert len(checks) == expected['successful_final_fits']
    with (ROOT / 'tables/confirmation_scores.csv').open(encoding='utf-8-sig') as f:
        published = {(r['case'], int(r['seed']), r['method']): float(r['rmse_original'])
                     for r in csv.DictReader(f)}
    assert len(published) == len(checks)
    for row in checks:
        assert np.isclose(row['rmse_original'], published[row['case'], row['seed'], row['method']],
                          rtol=1e-12, atol=1e-12)
    result = dict(archives=len(per_archive), result_records=len(trials),
                  trial_status_counts=statuses, verified_final_predictions=len(checks),
                  max_rmse_recompute_error=max(r['error_abs'] for r in checks),
                  all_hashes_verified=True, per_archive=per_archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    verify()
