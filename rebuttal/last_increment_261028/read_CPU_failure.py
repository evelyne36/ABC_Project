"""Read one archived N3 failure. No estimator fit or data/model mutation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import pathlib,sys,json,hashlib
import numpy as np
import capacity_core as core
old=pathlib.Path('/home/admin/crossgb_rebuttal_20261026');new=pathlib.Path('/home/admin/crossgb_fastlane_20261027');sys.path.insert(0,str(old/'rebuttal_baselines_20261025'));sys.path.insert(0,str(new/'rebuttal_execution_20261025/native/controlled/python-package'))
from baselines.protocol import load_protocol
import xgboost as xgb
base=old/'rebuttal_formal_20261026/results/N3_equal_search/cpu_activity__outer4__current_metagb';budget=json.loads((base/'budget_results.json').read_text());ci=budget['checkpoints'][-1]['selected_candidate_index'];folder=base/f'final_candidate{ci}';X,y,folds,manifest=load_protocol(old/'rebuttal_baselines_20261025/inputs/cpu_activity');fold=next(f for f in folds if f['outer_fold']==4);booster=xgb.Booster(model_file=str(folder/'model.ubj'));booster.set_param({'nthread':1});raw=json.loads(booster.save_raw(raw_format='json'))['learner'];trees=raw['gradient_booster']['model']['trees'];base_score=core.scalar(raw['learner_model_param']['base_score']);pred={s:booster.predict(xgb.DMatrix(X[fold[s]])) for s in ['train','test']}
with np.load(folder/'predictions.npz',allow_pickle=False) as a:diff=float(np.max(np.abs(a['prediction']-pred['test'])));assert np.array_equal(a['sample_id'],fold['test']) and np.allclose(a['y_true'],y[fold['test']])
assert diff<1e-5,('archived prediction mismatch',diff)
rmse={s:float(np.sqrt(np.mean((pred[s]-y[fold[s]])**2))) for s in pred};Xt=np.asarray(X[fold['train']],np.float32);Xe=np.asarray(X[fold['test']],np.float32);rt=core.crossgb_leaf_routes(Xt,trees);re=core.crossgb_leaf_routes(Xe,trees);errors=np.abs(pred['test']-y[fold['test']]);worst=np.argsort(errors)[-3:][::-1];details=[];coeff=[]
for tree in trees:
 coeff.extend(abs(float(tree['split_conditions'][i])) for i,l in enumerate(tree['left_children']) if l==-1)
for ix in worst:
 contrib=[]
 for ti,tree in enumerate(trees):
  leaf=int(re[ix,ti]);f=int(tree['metagb_leaf_features'][leaf]);w=float(tree['split_conditions'][leaf]);value=w*float(Xe[ix,f]) if f<Xe.shape[1] else w;contrib.append((abs(value),ti,leaf,f,w,value))
 top=[]
 for _,ti,leaf,f,w,value in sorted(contrib,reverse=True)[:3]:
  mask=rt[:,ti]==leaf;v=Xt[mask,f] if f<Xt.shape[1] else np.ones(int(mask.sum()));test_value=float(Xe[ix,f]) if f<Xe.shape[1] else 1.
  top.append(dict(tree=ti,leaf=leaf,feature=f,coefficient=w,contribution=value,train_leaf_support=int(mask.sum()),test_feature=test_value,train_leaf_feature_min=float(v.min()) if len(v) else None,train_leaf_feature_max=float(v.max()) if len(v) else None,outside_leaf_train_range=bool(len(v) and (test_value<v.min() or test_value>v.max()))))
 details.append(dict(test_row=int(fold['test'][ix]),truth=float(y[fold['test'][ix]]),prediction=float(pred['test'][ix]),top_contributions=top))
print(json.dumps(dict(scope='readonly original N3 CPU outer4 selected900s model; not N5 anchor',model_fit_calls=0,candidate=ci,archived_prediction_max_abs_difference=diff,rmse=rmse,target_test_min=float(y[fold['test']].min()),target_test_max=float(y[fold['test']].max()),absolute_leaf_coefficient_max=max(coeff),absolute_leaf_coefficient_p99=float(np.quantile(coeff,.99)),worst=details,interpretation='Descriptive failure localization only; no causal intervention or shared-root-cause claim.')))
