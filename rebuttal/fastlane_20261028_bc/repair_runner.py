"""Repair only the two existing S3 capacity probes; never fit an anchor."""
import concurrent.futures,datetime,hashlib,json,os,pathlib,signal,subprocess,sys,time
r=pathlib.Path('/home/admin/crossgb_fastlane_20261027');here=pathlib.Path(__file__).resolve().parent
expected=json.loads((here/'expected.json').read_text())['models'];out=r/'results/formal_s3_repair_v2';out.mkdir(parents=True,exist_ok=False)
cores=sorted(os.sched_getaffinity(0))
def one(pair):
 i,d=pair;seed=d['seed'];model=r/f'results/formal_followup_v1/S3_sparse/final/CrossGB/s{seed}';data=r/f'fastlane_data/S3_sparse__s{seed}.npz'
 for f,key in [(model/'model.json','local_model_sha256'),(data,'data_sha256')]:
  if hashlib.sha256(f.read_bytes()).hexdigest()!=d[key]:raise RuntimeError('Repair source mismatch: '+str(f))
 dest=out/'S3_sparse/capacity'/f's{seed}';dest.mkdir(parents=True)
 cmd=[sys.executable,str(here/'capacity.py'),'--root',str(r/'rebuttal_execution_20261025/native/controlled'),'--data',str(data),'--model-dir',str(model),'--output',str(dest)]
 with (dest/'process.log').open('w') as log:
  p=subprocess.Popen(['taskset','-c',str(cores[i])]+cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=300)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();code=124
 f=dest/'capacity.json';v=json.loads(f.read_text()) if f.exists() else {}
 return dict(seed=seed,exit_code=code,status=v.get('status'),replay=v.get('replay'),schema_version=v.get('schema_version'))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,enumerate(expected)))
(out/'complete.json').write_text(json.dumps(dict(rows=rows,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
