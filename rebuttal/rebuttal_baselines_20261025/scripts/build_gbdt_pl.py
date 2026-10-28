#!/usr/bin/env python3
"""Build pinned GBDT-PL with explicit platform compatibility; no training."""
from pathlib import Path
import concurrent.futures
import difflib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = next((ROOT / "vendor/gbdt_pl").glob("GBDT-PL-*"))
    work = ROOT / "build/gbdt_pl_source"
    if not work.exists():
        shutil.copytree(source, work, ignore=shutil.ignore_patterns("*.so", "*.pyc", "liblineargbm.py", "figures", "GBDT-PL"))
    # The build copy is generated, never a place for hand-edited source changes.
    for path in source.rglob("*"):
        if path.suffix in {".cpp", ".hpp", ".h"}:
            (work/path.relative_to(source)).write_text(path.read_text())
    # The upstream C API declares int return values but omits returns in these
    # functions. Define the success return; leave model construction unchanged.
    original = (source / "src/interface/interface.cpp").read_text()
    patched = original
    for function in ["CreateLinearGBMDataMat", "LinearGBMPrintBoosterConfig", "LinearGBMPredict", "LinearGBMBestIteration", "LinearGBMGetScoresPerIteration"]:
        start = patched.index("int " + function + "(")
        brace = patched.index("{", start)
        level, end = 1, brace + 1
        while level:
            level += (patched[end] == "{") - (patched[end] == "}")
            end += 1
        body = patched[brace:end]
        if "return " not in body:
            patched = patched[:end-1] + "    return 0;\n" + patched[end-1:]
    (work / "src/interface/interface.cpp").write_text(patched)
    differences = ["".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True), fromfile="upstream/interface.cpp", tofile="compat/interface.cpp"))]
    for path in (work/"src").rglob("*.cpp"):
        before = path.read_text()
        after = re.sub(r"num_threads\(\d+\)", "num_threads(1)", before)
        if path.name == "data_reader.cpp":
            # Keep the original two logical pipeline roles, but execute both
            # loop iterations on a single OpenMP worker in this benchmark.
            after = after.replace("num_threads(num_threads)", "num_threads(1)")
        if after != before:
            path.write_text(after)
            differences.append("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="upstream/"+str(path.relative_to(work)), tofile="compat/"+str(path.relative_to(work)))))
    (ROOT / "reports/gbdt_pl_compatibility.patch").write_text("".join(differences))
    mac = platform.system() == "Darwin"
    compiler = os.environ.get("CXX", "clang++" if mac else "g++")
    flags = ["-std=c++11", "-O2", "-fPIC", "-I"+str(ROOT/"compat"), "-I"+str(work/"include")]
    if mac:
        base = Path(sys.base_prefix)
        default_omp = base if (base/"lib/libomp.dylib").is_file() and (base/"include/omp.h").is_file() else Path("/opt/homebrew/opt/libomp")
        omp = Path(os.environ.get("GBDTPL_LIBOMP_PREFIX", str(default_omp)))
        flags += ["-Xpreprocessor", "-fopenmp", "-I"+str(omp/"include")]
        links = ["-L"+str(omp/"lib"), "-Wl,-rpath,"+str(omp/"lib"), "-lomp", "-framework", "Accelerate"]
        suffix = ".dylib"
    else:
        flags += ["-fopenmp"]
        links = ["-fopenmp", "-llapack", "-lblas"]
        suffix = ".so"
    if platform.machine().lower() in {"x86_64", "amd64"}:
        flags += ["-mavx2", "-mbmi2", "-mpopcnt"]
    sources = sorted(p for p in (work/"src").rglob("*.cpp") if p.name != "main.cpp")
    objects = []
    commands = []
    objdir = ROOT / "build/gbdt_pl_objects"
    objdir.mkdir(exist_ok=True)
    def compile_one(path):
        obj = objdir / (str(path.relative_to(work)).replace("/", "_") + ".o")
        command = [compiler, *flags, "-c", str(path), "-o", str(obj)]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(str(path.relative_to(work))+"\n"+completed.stderr)
        return obj, command, completed.stderr
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for obj, command, stderr in pool.map(compile_one, sources):
            objects.append(obj); commands.append(command)
            if stderr:
                with (ROOT/"reports/gbdt_pl_build_warnings.log").open("a") as output:
                    output.write(stderr)
    library = ROOT / "build" / ("liblineargbm"+suffix)
    command = [compiler, "-dynamiclib" if mac else "-shared", *map(str, objects), *links, "-o", str(library)]
    subprocess.run(command, check=True)
    commands.append(command)
    report = {"library":str(library.relative_to(ROOT)), "sha256":hashlib.sha256(library.read_bytes()).hexdigest(),
              "platform":platform.platform(), "machine":platform.machine(), "split_gain_and_leaf_fitting_logic_unchanged":True,
              "openmp_prefix":str(omp) if mac else "compiler_runtime",
              "compatibility":["explicit C API success returns", "platform LAPACK DGESV", "ARM scalar equivalents for Intel SIMD/BMI2", "serial AUC sort on macOS", "one OpenMP worker including upstream hardcoded prediction/reader regions"],
              "formal_timing_warning":"ARM scalar timings are not upstream x86/MKL performance; Linux build/runtime requires its own verification", "commands":commands}
    (ROOT/"reports/gbdt_pl_build.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="commands"}, indent=2))


if __name__ == "__main__":
    main()
