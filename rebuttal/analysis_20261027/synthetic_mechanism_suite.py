"""Matched mechanism stress tests; generated data are hypotheses, not claimed wins.

Uses NumPy only. All learners receive all X columns; truth is for diagnostics,
never an oracle input. A 72-case discovery grid is declared before fitting.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

FAMILIES = ('sparse_switch', 'dense_switch', 'offset_switch', 'step_control')
DEV_SEEDS = (31001, 31002, 31003)
CONFIRM_SEEDS = (41001, 41002, 41003, 41004, 41005)

def generate(family, n=600, d=32, seed=31001, regions=4, noise=.3):
    if family not in FAMILIES or d < 8 or regions not in (2, 4, 8) or n < 20:
        raise ValueError('Use a declared family, d>=8, regions in {2,4,8}, n>=20')
    rng = np.random.default_rng(seed)
    c = rng.uniform(-1., 1., size=n)
    z = rng.normal(size=(n, d-1))
    eps = rng.normal(scale=noise, size=n)
    X = np.column_stack((c, z)).astype(np.float32)
    region = np.minimum(((X[:, 0] + 1.) * regions / 2).astype(int), regions-1)
    sign = np.where(region % 2 == 0, 1., -1.)
    active = 1 + (region // 2) % (d-1)
    if family == 'step_control':
        signal = 2. * sign
        support = np.zeros((n, 1), dtype=np.int64) - 1
    else:
        k = 4 if family == 'dense_switch' else 1
        support = 1 + ((active[:, None]-1 + np.arange(k)) % (d-1))
        signal = 2. * sign * X[np.arange(n)[:, None], support].sum(axis=1) / np.sqrt(k)
        if family == 'offset_switch':
            # Match noiseless variance (=4 in the population) to the sparse case.
            signal = (signal + 2. * sign) / np.sqrt(2.)
    y = (signal + eps).astype(np.float32)
    truth = {'family':family, 'n':n, 'd':d, 'seed':seed, 'regions':regions,
             'noise_std':noise, 'gate_column':0, 'conditional_mean_signal_variance':4.,
             'active_predictor_count':4 if family=='dense_switch' else (0 if family=='step_control' else 1),
             'off_path_predictors_allowed_for_all_methods':True,
             'no_oracle_role_masks_supplied':True}
    return dict(X=X, y=y, signal=signal.astype(np.float32), region=region,
                active_support=support, noise=eps.astype(np.float32)), truth

def grid():
    return [{'family':f,'n':n,'d':d,'seed':s,'regions':4,'noise':.3}
            for f in FAMILIES for n in (600,2400) for d in (8,32,128) for s in DEV_SEEDS]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--materialize',action='store_true')
    a=p.parse_args(); cases=grid(); index=[]
    for case in cases:
        arrays,truth=generate(**case)
        assert arrays['X'].shape==(case['n'],case['d'])
        assert np.isfinite(arrays['X']).all() and np.isfinite(arrays['y']).all()
        np.testing.assert_allclose(arrays['y'],arrays['signal']+arrays['noise'],atol=2e-6,rtol=2e-6)
        duplicate,_=generate(**case)
        assert np.array_equal(arrays['X'],duplicate['X']) and np.array_equal(arrays['y'],duplicate['y'])
        # Every family's X and random noise match its positive/negative counterpart.
        counterpart,_=generate(**{**case,'family':'sparse_switch'})
        assert np.array_equal(arrays['X'],counterpart['X']) and np.array_equal(arrays['noise'],counterpart['noise'])
        ident=f"{case['family']}__n{case['n']}__d{case['d']}__seed{case['seed']}"
        row={'id':ident,**truth,'X_sha256':hashlib.sha256(arrays['X'].tobytes()).hexdigest(),
             'y_sha256':hashlib.sha256(arrays['y'].tobytes()).hexdigest()}
        if a.materialize:
            if a.output is None:p.error('--materialize requires --output')
            a.output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(a.output/(ident+'.npz'),**arrays)
        index.append(row)
    report={'status':'generator_validated_no_model_fits','cases':index,'discovery_seeds':DEV_SEEDS,
            'reserved_confirmation_seeds_not_evaluated':CONFIRM_SEEDS,
            'claims':'No expected winner is certified. Dense and offset cases can favour affine neighbours; report all cases.',
            'note':'sparse_switch extends existing regime_slope/feature_switch mechanisms; it is not presented as a novel generator.'}
    if a.output is not None:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'suite_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(index),'model_fits':0,'data_materialized':a.materialize}))

if __name__=='__main__':main()
