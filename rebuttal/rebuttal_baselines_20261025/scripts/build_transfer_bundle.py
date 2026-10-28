#!/usr/bin/env python3
"""Package source, frozen inputs and configs; exclude local environments/binaries."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=[]
    for path in ROOT.rglob("*"):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if any(part in {".venv","__pycache__",".pytest_cache","build","downloads"} for part in relative.parts):continue
        if relative.parts[0]=="vendor":
            if path.suffix in {".py",".cpp",".hpp",".h"}:
                if b"\x00" in path.read_bytes()[:1024]:continue
            elif path.name not in {"CMakeLists.txt","README.md","Readme.md","LICENSE","LICENSE.md"}:
                continue
        elif relative.parts[0]=="reports":
            if path.name not in {"sources.json","setup_status.json","gbdt_pl_compatibility.patch"}:continue
        elif relative.parts[0] not in {"baselines","compat","configs","inputs","scripts","tests"}:
            if path.name not in {"README_本机适配与实验入口.md","requirements.txt","WORKLOG.md"}:continue
        paths.append(path)
    paths=sorted(paths)
    bundle=ROOT/"baseline_linux_bundle.tar.gz"
    with tarfile.open(bundle,"w:gz") as output:
        for path in paths:output.add(path,arcname=str(Path(ROOT.name)/path.relative_to(ROOT)),recursive=False)
    report={"bundle":bundle.name,"sha256":hashlib.sha256(bundle.read_bytes()).hexdigest(),"bytes":bundle.stat().st_size,
            "files":[str(p.relative_to(ROOT)) for p in paths],"excluded":"Mac environment/build/native binaries/downloads and upstream example datasets"}
    (ROOT/"reports/transfer_bundle.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="files"},indent=2))


if __name__=="__main__":main()
