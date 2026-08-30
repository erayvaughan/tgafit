# TGA Kinetics Fitter

Automated kinetic analysis of thermogravimetric (TGA) data: give it an
instrument export, get fitted multi-step decomposition kinetics with
validation figures.

## Where things are

| Folder | Contents |
|---|---|
| **`tgafit_main/`** | **The tool.** Start here — see its [README](tgafit_main/README.md). Entry point: `analyze.py`. |
| `tools/` | `prepare_tga.py` — instrument-export reader / problem generator (used by `analyze.py`). |
| `reports/` | Written reports and slide decks. Local only, not tracked by git. |

## Quick start

```bash
cd tgafit_main
.CondaPkg\.pixi\envs\default\python.exe analyze.py "your_run.txt"
```

Verify the install without any data of your own:

```bash
cd tgafit_main
.CondaPkg\.pixi\envs\default\python.exe analyze.py --selftest
```

This runs a synthetic twin experiment. The code generates a curve from known
kinetics, fits it back and reports how close it got.

Full documentation: [tgafit_main/README.md](tgafit_main/README.md) and
[tgafit_main/CODE_GUIDE.md](tgafit_main/CODE_GUIDE.md).

## What is not in the repository

The following are kept out by `.gitignore`, so a fresh clone contains only the
publishable code:

- `tgafit_main/raw_exports/` for raw instrument files. Put your own exports here.
- `tgafit_main/data/` and `tgafit_main/problems/` for confidential conditioned
  measurements and fitted problem definitions.
- `tgafit_main/results/` for every figure and table the pipeline produces. All
  of it comes back by rerunning the analysis.
- `reports/` for the written reports and slide decks, which are large binaries
  holding unpublished results.
