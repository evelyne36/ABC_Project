#!/usr/bin/env python3
"""Restricted dispatch to the five user-authorized rebuttal hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HOSTS = ["cpu4_pool.zjuici.com", "cpu2_pool.zjuici.com", "cpu1_pool.zjuici.com", "cpu_Calculator", "TabularData_Processor"]
WORKDIR = "/home/admin/crossgb_rebuttal_20261026"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=6"]


def ssh(host, program, timeout=60):
    return subprocess.run(SSH + [host, "python3 -c " + shlex.quote(program)], capture_output=True, text=True, timeout=timeout)


def inspect(host):
    program = """import json,os,platform,shutil,importlib.metadata as m
from pathlib import Path
names=['numpy','scipy','pandas','scikit-learn','numba','networkx','threadpoolctl','pytest','cmake']
versions={}
for n in names:
 try: versions[n]=m.version(n)
 except m.PackageNotFoundError: versions[n]=None
paths=[Path('/usr/bin/python3.11'),Path('/usr/bin/python3.12'),Path('/usr/bin/python3.13'),Path('/home/admin/miniconda3/bin/python'),Path('/home/admin/anaconda3/bin/python'),Path('/opt/conda/bin/python')]
for prefix in ['miniconda3','anaconda3','.conda']:
 paths+=list((Path.home()/prefix/'envs').glob('*/bin/python'))
print(json.dumps({'system':platform.system(),'machine':platform.machine(),'hostname':platform.node(),'home':str(Path.home()),'python':platform.python_version(),'affinity':sorted(os.sched_getaffinity(0)),'meminfo':Path('/proc/meminfo').read_text().splitlines()[:3],'loadavg':Path('/proc/loadavg').read_text().strip(),'versions':versions,'python_candidates':[str(p) for p in paths if p.exists()],'tools':{n:shutil.which(n) for n in ['cmake','g++','rsync','taskset','pip3','conda','uv']},'workdir_exists':Path('/home/admin/crossgb_rebuttal_20261026').exists()}))
"""
    result = ssh(host, program)
    return {"host": host, "exit_code": result.returncode, "details": json.loads(result.stdout) if result.returncode == 0 else None, "error": result.stderr}


def environments(host):
    program = """import json,subprocess
from pathlib import Path
paths=[Path('/home/admin/anaconda3/bin/python')]+list(Path('/home/admin/anaconda3/envs').glob('*/bin/python'))
code=\"import json,platform,importlib.metadata as m; names=['numpy','scipy','pandas','scikit-learn','numba','networkx','threadpoolctl','pytest']; pkgs={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}; print(json.dumps({'python':platform.python_version(),'packages':{n:pkgs.get(n) for n in names}}))\"
rows=[]
for path in paths:
 r=subprocess.run([str(path),'-c',code],capture_output=True,text=True,timeout=15)
 rows.append({'path':str(path),'versions':json.loads(r.stdout) if r.returncode==0 else None,'error':r.stderr[-300:]})
print(json.dumps(rows))
"""
    result = ssh(host, program, timeout=90)
    return {"host": host, "exit_code": result.returncode, "details": json.loads(result.stdout) if result.returncode == 0 else None, "error": result.stderr}


def deploy(host):
    package = ROOT / "rebuttal_formal_20261026/formal_linux_bundle.tar.gz"
    create = "from pathlib import Path; import json; p=Path(" + repr(WORKDIR) + "); marker=p/'deployment_marker.json'; assert not p.exists() or marker.exists(), 'Refuse to overwrite an unrelated directory'; p.mkdir(exist_ok=True); marker.write_text(json.dumps({'task':'crossgb_rebuttal_20261026'})); print(str(p))"
    result = ssh(host, create)
    if result.returncode: return {"host": host, "phase": "create", "exit_code": result.returncode, "error": result.stderr}
    transfer = subprocess.run(["rsync", "-az", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=5 -o ServerAliveCountMax=6", str(package), host + ":" + WORKDIR + "/formal_linux_bundle.tar.gz"], capture_output=True, text=True, timeout=240)
    if transfer.returncode: return {"host": host, "phase": "transfer", "exit_code": transfer.returncode, "error": transfer.stderr}
    program = """import tarfile,json,hashlib
from pathlib import Path
p=Path(""" + repr(WORKDIR) + """)
with tarfile.open(p/'formal_linux_bundle.tar.gz') as t:
 for member in t.getmembers():
  assert not member.issym() and not member.islnk() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
 t.extractall(p)
with tarfile.open(p/'rebuttal_baselines_20261025/baseline_linux_bundle.tar.gz') as t:
 for member in t.getmembers():
  assert not member.issym() and not member.islnk() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
 t.extractall(p)
manifest=json.loads((p/'rebuttal_formal_20261026/deployment_manifest.json').read_text())
errors=[r['path'] for r in manifest['files'] if hashlib.sha256((p/r['path']).read_bytes()).hexdigest()!=r['sha256']]
print(json.dumps({'workdir':str(p),'verified_files':len(manifest['files']),'hash_errors':errors}))
assert not errors
"""
    result = ssh(host, program, timeout=120)
    return {"host": host, "phase": "deploy", "exit_code": result.returncode, "details": result.stdout, "error": result.stderr}


def build(host):
    program = """import subprocess,json,os,hashlib
from pathlib import Path
p=Path(""" + repr(WORKDIR) + """)
manifest=json.loads((p/'rebuttal_formal_20261026/deployment_manifest.json').read_text())
assert all(hashlib.sha256((p/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['files']), 'Deployment input hash mismatch'
pidfile=p/'native_build.pid'
if pidfile.exists():
 try:
  os.kill(int(pidfile.read_text()),0); print(json.dumps({'already_running_pid':int(pidfile.read_text())})); raise SystemExit(0)
 except ProcessLookupError: pass
log=(p/'native_build.log').open('w')
process=subprocess.Popen(['bash','-c','bash rebuttal_formal_20261026/scripts/build_controlled_linux.sh; code=$?; echo $code > native_build.exit; exit $code'],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(process.pid));print(json.dumps({'started_build_pid':process.pid,'workdir':str(p)}))
"""
    result = ssh(host, program)
    return {"host": host, "phase": "build", "exit_code": result.returncode, "details": result.stdout, "error": result.stderr}


def status(host):
    program = """import json
from pathlib import Path
p=Path(""" + repr(WORKDIR) + """)
print(json.dumps({'workdir_exists':p.exists(),'bundle_bytes':(p/'formal_linux_bundle.tar.gz').stat().st_size if (p/'formal_linux_bundle.tar.gz').exists() else None,'source_ready':(p/'rebuttal_execution_20261025/native/controlled/CMakeLists.txt').exists(),'build_exit':(p/'native_build.exit').read_text() if (p/'native_build.exit').exists() else None,'build_tail':(p/'native_build.log').read_text()[-1800:] if (p/'native_build.log').exists() else None}))
"""
    result = ssh(host, program)
    return {"host": host, "phase": "status", "exit_code": result.returncode, "details": result.stdout, "error": result.stderr}


def connectivity(host):
    result = subprocess.run(SSH + ["-vv", host, "true"], capture_output=True, text=True, timeout=30)
    return {"host": host, "phase": "connectivity", "exit_code": result.returncode,
        "diagnostic_tail": result.stderr[-4200:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["inspect", "environments", "deploy", "build", "status", "connectivity"], required=True)
    parser.add_argument("--hosts", choices=HOSTS, nargs="+", default=HOSTS)
    args = parser.parse_args()
    function = {"inspect": inspect, "environments": environments, "deploy": deploy, "build": build, "status": status, "connectivity": connectivity}[args.phase]
    def guarded(host):
        try: return function(host)
        except subprocess.TimeoutExpired as error: return {"host": host, "phase": args.phase, "exit_code": "timeout", "error": str(error)}
    with ThreadPoolExecutor(max_workers=1 if args.phase == "deploy" else 5) as pool:
        reports = []
        for report in pool.map(guarded, args.hosts):
            reports.append(report); print(json.dumps(report), flush=True)
    path = ROOT / ("rebuttal_formal_20261026/results/remote_" + args.phase + ".json")
    path.write_text(json.dumps(reports, indent=2))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path.with_name(path.stem + "_" + stamp + ".json").write_text(json.dumps(reports, indent=2))
    raise SystemExit(0 if all(r["exit_code"] == 0 for r in reports) else 2)


if __name__ == "__main__": main()
