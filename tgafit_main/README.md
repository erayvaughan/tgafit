# TGA Kinetics Fitter

Give it a thermogravimetric (TGA) export — get fitted multi-step decomposition
kinetics with validation figures. No per-material code changes.

## Quick start (one command)

```bash
python analyze.py "C:/path/to/your_run.xlsx"
```

Accepts TA TRIOS `.xls` / `.xlsx` exports or plain `.csv` / `.txt` with
temperature + weight columns. Output lands in `results/`:

| File | Contents |
|---|---|
| `<name>_prep.png` | diagnostic: conditioned curve, DTG, detected steps |
| `<name>_fit_report.png` | data vs fitted model + residuals (RMSE, R²) |
| `<name>_paper_style.png` | mass loss + per-component curves + DTG panel (°C) |
| `<name>_parameter_comparison.csv` | fitted log₁₀A, E, n per reaction step |
| `problems/<name>.toml` | the auto-generated, editable problem definition |

## How it works

1. **Prepare** (`tools/prepare_tga.py`): finds the heating ramp in the export,
   measures the heating rate from the data, normalizes mass, detects mass-loss
   steps as DTG peaks, and derives physics-based initial guesses
   (E from peak width, A from peak position). Writes a clean CSV + problem TOML.
2. **Fit** (`solve.jl`, Julia): builds a parallel pseudo-component reaction
   model (one Arrhenius reaction per detected step + inert residue) with
   Catalyst.jl, and estimates the kinetic parameters against the measured curve
   with PEtab.jl / LBFGS.
3. **Report** (`fit_report.jl`): RMSE / MAE / max error / R² against the data,
   plus publication-style figures (axes in °C).

Steps 2–3 can also be run individually:

```bash
julia --project=. solve.jl problems/<name>.toml            # fit
julia --project=. solve.jl problems/<name>.toml --forward 5 0.21   # forward sim only
julia --project=. fit_report.jl problems/<name>.toml       # metrics + figures
```

## The model

Parallel pseudo-component scheme (standard for TGA of unknown materials):
each detected mass-loss step i becomes an independent reaction

    P_i -> Gas      dα/dt = A_i · exp(−E_i / RT) · (1−α)^(n_i)

plus an inert `Residue` fixed at the measured final plateau. Component names
(`P1`, `P2`, …) are labels, not chemistry — rename them in the TOML if the
material composition is known.

Multi-condition fitting is supported: list several experiments (different
heating rates / atmospheres, one CSV each) in the TOML's `[experiments]`
section and they are fit jointly — recommended to break the A–E correlation
inherent to single-heating-rate data.

## Good to know

- **Legacy `.xls` exports truncate at 65,536 rows.** If your run is longer
  (e.g. 10 Hz sampling to 1000 °C), the heating data gets cut mid-ramp — the
  tool warns when this happens. Export as `.xlsx`/`.csv` instead.
- Kinetics are computed in Kelvin internally (Arrhenius requires absolute
  temperature); all plots display °C.
- First Julia run precompiles packages and is slow (minutes); later runs are faster.

## Dependencies

Julia (Catalyst.jl, PEtab.jl, OrdinaryDiffEq.jl, Plots.jl — see `Project.toml`;
run `julia --project=. -e "using Pkg; Pkg.instantiate()"` once) and Python 3
with `numpy`, `pandas`, `xlrd`, `openpyxl`, `matplotlib`.
