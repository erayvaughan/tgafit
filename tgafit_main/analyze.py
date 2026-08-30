#!/usr/bin/env python
"""
One command: TGA export in -> fitted kinetics + figures out.

    python analyze.py "C:/path/to/your_run.xlsx" [more files...]

Accepts TA TRIOS .xls/.xlsx exports or plain .csv/.txt with temperature+weight
columns. For each file it:

  1. finds the heating ramp, converts units, detects mass-loss steps (DTG),
     derives physics-based initial guesses, writes data/<name>.csv +
     problems/<name>.toml + a diagnostic plot            (prepare_tga)
  2. fits the multi-step Arrhenius kinetics to the data   (PEtab / LBFGS)
  3. writes metrics (RMSE / R^2) and figures:
       results/<name>_fit_report.png    data vs fit + residuals
       results/<name>_paper_style.png   mass loss + species + DTG (deg C)
       results/<name>_parameter_comparison.csv   fitted parameters

Everything is derived from the file itself - no per-material code changes.
"""
import os, shutil, subprocess, sys

# =============================================================================
# EDIT THIS LINE: file(s) to analyze when you run  `python analyze.py`  with no
# arguments. A bare filename is looked up in this folder; full paths work too.
# =============================================================================
FILES_TO_ANALYZE = [
    "fiber_air_2.5 c per min.xlsx",
    "fiber_air_5 c per min.xlsx",
    "fiber_air_10 c per min.xlsx",
    "fiber_air_20 c per min.xlsx",
]

# Optional JOINT fit: if set (e.g. "af340_joint") and more than one file is
# listed above, ALL files are additionally fit TOGETHER as one multi-condition
# problem with shared kinetics (use for the same material at several heating
# rates / atmospheres). Each file still gets its own individual results first.
JOINT_NAME = "fiber_joint"

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
sys.path.insert(0, TOOLS)
import prepare_tga  # noqa: E402

JULIA_FALLBACK = os.path.expanduser(
    r"~\AppData\Local\Microsoft\WindowsApps\julia.exe")

def julia_exe():
    return shutil.which("julia") or JULIA_FALLBACK

JL_TEMPLATE = r'''
using Logging; Logging.disable_logging(Logging.Warn)
include("fit_report.jl")           # loads the engine too
spec = load_problem("{toml}")
solve_problem(spec)
for f in ["parameter_comparison.csv", "validation_plot.png",
          "all_experiments.png", "measurements.csv"]
    src = joinpath("results", f); dst = joinpath("results", spec.name * "_" * f)
    isfile(src) && mv(src, dst; force = true)
end
fit_report("{toml}")
'''

def _fit_and_report(toml_path):
    toml_rel = os.path.relpath(toml_path, HERE).replace("\\", "/")
    print("\n-- fitting kinetics (this takes a few minutes) --")
    code = JL_TEMPLATE.format(toml=toml_rel)
    r = subprocess.run([julia_exe(), "--project=.", "-e", code], cwd=HERE)
    name = os.path.splitext(os.path.basename(toml_path))[0]
    if r.returncode != 0:
        print(f"[!] fit failed for {name} (exit {r.returncode})")
        return False
    print(f"\nDONE - results for '{name}' in results/:")
    for suffix in ("_fit_report.png", "_paper_style.png", "_parameter_comparison.csv"):
        f = os.path.join(HERE, "results", name + suffix)
        print(f"  {'OK ' if os.path.isfile(f) else '?? '}{os.path.relpath(f, HERE)}")
    return True

def analyze(path):
    print("=" * 70)
    print(f"ANALYZING: {path}")
    print("=" * 70)
    toml_path = prepare_tga.emit(path, HERE)
    return _fit_and_report(toml_path)

def analyze_joint(paths, name):
    toml_path = prepare_tga.emit_joint(paths, HERE, name)
    return _fit_and_report(toml_path)

def resolve(name):
    """Bare filenames are searched in this folder, raw_exports/, then Downloads."""
    if os.path.isfile(name):
        return name
    for base in (HERE, os.path.join(HERE, "raw_exports"), os.path.expanduser("~/Downloads")):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    raise SystemExit(f"file not found: {name} (looked in {HERE}, raw_exports/ and Downloads)")

def selftest():
    """Synthetic round-trip: generate data from known parameters, refit, and
    require the mean recovery error to stay below 10%. One-command check that
    the model/estimation chain still works after any code change."""
    code = r'''
using Logging; Logging.disable_logging(Logging.Warn)
include("src/engine/engine.jl")
using Statistics
r = solve_problem(load_problem("tests/selftest.toml"))
err = mean(r.comparison.Pct_Error)
println("SELFTEST mean recovery error: ", round(err, digits=2), "%")
if err < 10.0
    println("SELFTEST PASS")
else
    println("SELFTEST FAIL (threshold 10%)"); exit(1)
end
'''
    r = subprocess.run([julia_exe(), "--project=.", "-e", code], cwd=HERE)
    return r.returncode == 0

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    targets = sys.argv[1:] or FILES_TO_ANALYZE
    if not targets:
        raise SystemExit(__doc__)
    paths = [resolve(p) for p in targets]
    ok = all(analyze(p) for p in paths)          # individual results per file
    if JOINT_NAME and len(paths) > 1:            # plus one joint multi-rate fit
        ok = analyze_joint(paths, JOINT_NAME) and ok
    sys.exit(0 if ok else 1)
