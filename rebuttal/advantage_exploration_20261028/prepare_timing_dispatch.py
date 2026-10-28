import json
from build_dispatch import HERE,package,sha
frozen=json.loads((HERE/'confirmation_freeze.json').read_text());jobs=[];files={}
for kind,case,seed in [('synthetic','A06_two_sparse_modules',111001),('real','gas_turbine_nox',113001)]:
 entries=json.loads((HERE/kind/'confirmation/manifest.json').read_text())['entries'];e=next(e for e in entries if e['seed']==seed)
 f=HERE/kind/'confirmation'/(e.get('data_npz') or e['npz']);assert sha(f)==e['npz_sha256'];remote='advantage_inputs/confirmation/'+kind+'/'+f.name;files[remote]=f
 # Rotate method order across repetitions to avoid a single fixed ordering.
 methods=frozen['methods']+['LeafOnly']
 for repeat in range(3):
  for method in methods[repeat:]+methods[:repeat]:
   choice=frozen['choices'][case]['CrossGB' if method=='LeafOnly' else method];jid=f'{case}/{method}/s{seed}/r{repeat:02}'
   jobs.append(dict(id=jid,case=case,kind=kind,method=method,candidate=choice['candidate'],seed=seed,repeat=repeat,params=choice['params'],data=remote,relative_output=jid,evaluate_test=True,timeout_seconds=300))
d=package('exclusive_timing',jobs,files,core_slot=0,wall_budget_seconds=1200,host_assignment={'A06_two_sparse_modules':'cpu1_pool.zjuici.com','gas_turbine_nox':'cpu6_pool.zjuici.com'});print(json.dumps(dict(jobs=len(jobs),dispatch=str(d))))
