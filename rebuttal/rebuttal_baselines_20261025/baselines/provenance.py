from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def code_fingerprint(directory):
    directory=Path(directory);records={}
    for path in sorted(directory.rglob("*")):
        if path.suffix not in {".py",".cpp",".hpp",".h"} or not path.is_file():
            continue
        content=path.read_bytes()
        if b"\x00" in content[:1024]:
            continue  # Upstream GBDT-PL ships an ELF file with a .py extension.
        records[str(path.relative_to(directory))]=hashlib.sha256(content).hexdigest()
    digest=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return digest,records


def source_record(method):
    sources=json.loads((ROOT/"reports/sources.json").read_text())
    record=next(r for r in sources if r["method"]==method)
    actual,_=code_fingerprint(ROOT/record["extracted_path"])
    if actual!=record["code_fingerprint"]:
        raise RuntimeError("Pinned author code changed: "+method)
    return record


def adapter_fingerprint():
    records={}
    for folder in [ROOT/"baselines",ROOT/"compat"]:
        value,_=code_fingerprint(folder);records[folder.name]=value
    for filename in ["run_task.py","run_nested.py","build_gbdt_pl.py"]:
        records["scripts/"+filename]=hashlib.sha256((ROOT/"scripts"/filename).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
