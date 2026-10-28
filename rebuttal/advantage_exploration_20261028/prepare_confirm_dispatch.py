import json
from build_dispatch import HERE,package,sha
frozen=json.loads((HERE/'confirmation_freeze.json').read_text());jobs=[];files={}
for kind,case in [('synthetic','A06_two_sparse_modules'),('real','gas_turbine_nox')]:
 entries=json.loads((HERE/kind/'confirmation/manifest.json').read_text())['entries']
 for method in frozen['methods']+['LeafOnly']:
  choice=frozen['choices'][case]['CrossGB' if method=='LeafOnly' else method]
  for e in entries:
   f=HERE/kind/'confirmation'/(e.get('data_npz') or e['npz']);assert sha(f)==e['npz_sha256'];remote='advantage_inputs/confirmation/'+kind+'/'+f.name;files[remote]=f;seed=e['seed'];jid=f'{case}/{method}/s{seed}'
   jobs.append(dict(id=jid,case=case,kind=kind,method=method,candidate=choice['candidate'],seed=seed,params=choice['params'],data=remote,relative_output=jid,evaluate_test=True,save_model=method=='CrossGB',timeout_seconds=300))
d=package('confirmation',jobs,files,core_slot=0,wall_budget_seconds=1200);print(json.dumps(dict(jobs=len(jobs),dispatch=str(d))))
