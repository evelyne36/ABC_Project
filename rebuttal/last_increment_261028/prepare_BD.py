"""Prepare fixed, paired sensitivity inputs; no model-based data selection."""
from pathlib import Path
import json,hashlib,importlib.util,sys,datetime
import numpy as np
from build_dispatch import HERE,package,write,sha
R=HERE.parent;OLD=R/'advantage_exploration_20261028';F=json.loads((OLD/'confirmation_freeze.json').read_text());entries=[];jobs=[];files={};B=HERE/'inputs/B';B.mkdir(parents=True,exist_ok=False)
for seed in range(121001,121006):
 X=np.random.default_rng(np.random.SeedSequence([seed,261028,1])).normal(size=(896,32));eps=np.random.default_rng(np.random.SeedSequence([seed,261028,2])).normal(scale=.3,size=896);signal=np.sqrt(2)*(np.where(X[:,0]>=0,1.,-1.)*X[:,2]+np.where(X[:,1]>=0,1.,-1.)*X[:,3]);y=signal+eps
 for n in [144,288]:
  for d in [16,32]:
   case=f'B_train{n}_d{d}';name=case+'__s'+str(seed);p=B/(name+'.npz');train=np.arange(n);val=np.arange(288,384);test=np.arange(384,896)
   np.savez_compressed(p,X=X[:,:d].astype(np.float32),train=train,val=val,test=test,task_type=np.asarray('regression'),y_train=y[train],y_val=y[val],y_test=y[test]);meta=dict(case=case,seed=seed,kind='synthetic',preprocessing={'y_scale':1.},train_n=n,d=d,split_sizes=dict(train=n,val=96,test=512),signal='sqrt(2)(sign(x0)x2+sign(x1)x3)',noise_sd=.3,paired_master=True);write(p.with_suffix('.json'),meta);remote='last_increment_inputs/B/'+p.name;files[remote]=p
   entries.append(dict(case=case,seed=seed,data=str(p.relative_to(HERE)),metadata=str(p.with_suffix('.json').relative_to(HERE)),sha256=sha(p),kind='synthetic'))
   for method in ['CrossGB','LinXGBoost','PL','LGB_linear']:
    choice=F['choices']['A06_two_sparse_modules'][method];jid=f'{case}/{method}/s{seed}'
    jobs.append(dict(id=jid,case=case,method=method,candidate=choice['candidate'],seed=seed,params=choice['params'],data=remote,relative_output=jid,evaluate_test=True,timeout_seconds=180))
package('B',jobs,files,core_slot=0,wall_budget_seconds=1500,host_assignment={e['case']:'TabularData_Processor' for e in entries});write(HERE/'B_inputs.json',entries)
spec=importlib.util.spec_from_file_location('realdev',OLD/'real/prepare_real.py');dev=importlib.util.module_from_spec(spec);spec.loader.exec_module(dev);prior=dev.prior_runtime();X,y,groups,fixed,source,protocol=dev.load_task('gas_turbine_nox',prior)
D=HERE/'inputs/D';D.mkdir(parents=True,exist_ok=False);jobs=[];files={};entries=[]
for e in json.loads((OLD/'real/confirmation/manifest.json').read_text())['entries']:
 seed=e['seed'];oldp=OLD/'real/confirmation'/e['npz'];metaold=json.loads((OLD/'real/confirmation'/e['metadata']).read_text())
 with np.load(oldp,allow_pickle=False) as a:small=a['train'];val=a['val'];test=a['test'];pool=a['train_uncapped']
 assert np.array_equal(pool,fixed[0]) and np.array_equal(val,fixed[1]) and np.array_equal(test,fixed[2]);remaining=np.setdiff1d(pool,small);extra=np.random.default_rng(np.random.SeedSequence([seed,261028,2000])).choice(remaining,1400,replace=False);train=np.sort(np.r_[small,extra]);assert len(train)==2000 and np.isin(small,train).all()
 Z,target,pre=prior.standardize(X,y,train);case='gas_turbine_nox_n2000';name=case+'__s'+str(seed);p=D/(name+'.npz');np.savez_compressed(p,X=np.asarray(Z,np.float32),train=train,val=val,test=test,task_type=np.asarray('regression'),y_train=np.asarray(target[train],float),y_val=np.asarray(target[val],float),y_test=np.asarray(target[test],float));meta=dict(case=case,seed=seed,kind='real',preprocessing=pre,source=source,split_sizes=dict(train=2000,val=len(val),test=len(test)),nested_old_train600=True,old_input_sha256=sha(oldp),protocol=protocol,frozen_parameter_source=str(OLD/'confirmation_freeze.json'));write(p.with_suffix('.json'),meta);remote='last_increment_inputs/D/'+p.name;files[remote]=p;entries.append(dict(case=case,seed=seed,data=str(p.relative_to(HERE)),metadata=str(p.with_suffix('.json').relative_to(HERE)),sha256=sha(p),kind='real'))
 for method in F['methods']:
  choice=F['choices']['gas_turbine_nox'][method];jid=f'{case}/{method}/s{seed}'
  jobs.append(dict(id=jid,case=case,method=method,candidate=choice['candidate'],seed=seed,params=choice['params'],data=remote,relative_output=jid,evaluate_test=True,timeout_seconds=180))
package('D',jobs,files,core_slot=0,wall_budget_seconds=1200);write(HERE/'D_inputs.json',entries);write(HERE/'BD_freeze.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),B=dict(fits=80,seeds=list(range(121001,121006)),train_counts=[144,288],dimensions=[16,32],val_count=96,test_count=512,fixed_params='A06 existing frozen recipe, no new HPO'),D=dict(fits=30,training=2000,old_training=600,nested_train=True,shared_time_holdout=True,fixed_params='NOX existing frozen recipe, no new HPO')));print('Prepared B80 and D30 with fixed parameter recipes')
