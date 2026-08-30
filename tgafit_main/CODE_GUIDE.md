# Code Guide — Learn the TGA Kinetics Fitter in Depth

This document walks through every file and every function, in execution order,
with the reasoning behind each decision — the goal is that you could rewrite
the program from scratch after studying it. Read it next to the source files.

---

## 0. The big picture

The program answers one question: *given a measured TGA mass-loss curve, what
are the Arrhenius parameters (A, E, n) of each decomposition step?* That is an
inverse problem, and the code is organized as three stages connected by files:

```
export file  ──[Python: prepare]──►  data/<name>.csv  +  problems/<name>.toml
problems/<name>.toml ──[Julia: solve]──►  fitted parameters (CSV)
fitted parameters ──[Julia: report]──►  metrics + figures
```

Python does stage 1 (pandas is unbeatable at reading messy spreadsheets);
Julia does stages 2–3 (fast stiff-ODE integration and automatic
differentiation, which the optimizer needs). The two never call each other —
they only read files the other wrote. That means you can debug each half in
isolation, and a problem TOML is a complete, self-contained record of a fit.

---

## 1. `analyze.py` — the driver (~100 lines)

Purely orchestration; contains no science. Top to bottom:

- **`FILES_TO_ANALYZE` / `JOINT_NAME`** — the only things a user edits.
- **`julia_exe()`** — finds Julia via `shutil.which`, falling back to the
  Windows Store alias path. Needed because this machine has no `julia` on a
  standard PATH location for subprocesses.
- **`JL_TEMPLATE`** — a Julia one-liner template. It includes `fit_report.jl`
  (which itself includes the whole engine), calls `solve_problem`, renames the
  four generically-named outputs (`parameter_comparison.csv`, …) to
  `<name>_`-prefixed copies so different materials don't overwrite each other,
  then calls `fit_report`. Doing solve+report in ONE Julia process matters:
  Julia startup + package load costs ~1–2 minutes, so you only want to pay it
  once per problem.
- **`_fit_and_report(toml)`** — runs that template via `subprocess.run` and
  checks the three expected output files exist afterwards. The exit code of
  the subprocess is the success signal.
- **`analyze(path)`** — prepare + fit + report for one file.
- **`analyze_joint(paths, name)`** — same but via `emit_joint`.
- **`resolve(name)`** — bare filenames are searched in the project folder,
  then in `~/Downloads`. Convenience only.
- **`__main__`** — CLI args override `FILES_TO_ANALYZE`; after all individual
  analyses, the joint one runs if `JOINT_NAME` is set and >1 file given.

---

## 2. `tools/prepare_tga.py` — from raw export to fitting problem

This is the file with the most "street smarts". Four stages.

### 2.1 Reading (`_numeric_block`, `read_export`, `_read_lims_txt`, `_is_lims`)

**`_numeric_block(df)`** — given a raw sheet as a DataFrame with no header,
find the data table inside it. It scans the first 30 rows for a row whose
cells contain "temperature" and something starting with "weight" (or
containing "mass") — that's the header row. Column indexes for Time /
Temperature / Weight are taken from that row. **The row right below the
header is checked for a time unit** (`s` or `min`) — TA exports differ
(.xls uses seconds, .xlsx uses minutes) and getting this wrong once made the
computed heating rate wrong by a factor of 60. Everything below is coerced
with `pd.to_numeric(errors="coerce")` and rows where T or W is NaN are
dropped: that automatically removes stray text rows. Returns `(T, W, t,
time_unit)` or `None` if the sheet has no usable table.

**`_read_lims_txt(path)`** — the TRIOS "LIMS" text export is not a table but
a sequence of `[Step]` blocks, each with a `Variables` line (column names), a
`Units` line, and thousands of `Data point<TAB>...` lines. The parser walks
the file line by line, keeps state (current step name, column indexes for
Time/Temperature/Weight), and flushes a completed block into the same
candidate format the Excel path produces. Time is converted min→s using the
Units line. Why line-by-line instead of pandas? The blocks have different
column counts than the metadata lines around them, which breaks table
readers.

**`read_export(path)`** — dispatch + selection:
1. Route to the LIMS parser (`_is_lims` peeks at the first 200 chars for the
   `[Version]`/`LIMS export` signature), the Excel reader (every sheet is
   tried), or the delimited-text fallback (tries `,`, `;`, tab).
2. From all candidate blocks, **keep the one with the largest temperature
   RISE** — that's the heating ramp; cooling segments rise negatively and
   lose.
3. Unit handling: temperatures that look like Celsius (`max < 1500 and
   min < 200`) get +273.15. Weight is divided by its first value — works
   identically for % or mg exports.
4. **β is measured**, not assumed: linear regression (`np.polyfit`) of T
   against t, ×60 → K/min. If the file had no units row and β comes out
   > 150 K/min (no real TGA runs that fast), it assumes minutes and warns
   loudly — a heuristic of last resort, never silent.
5. **Truncation detection:** legacy .xls sheets hold at most 65,536 rows. If
   a .xls block has ≥65,530 rows, the run was almost certainly cut mid-ramp
   → warn.
6. **Measured steady state:** if another candidate block is a *cooling*
   segment whose maximum temperature exceeds the heating data's end by >10 K,
   its first points (within 3 K of its max T) are averaged → that is a genuine
   measurement of the end-of-run residual mass (the sample is inert while
   cooling). Stored as `steady=(T_K, mass)`.

### 2.2 Conditioning (`condition`, `smooth`)

`condition(T, mass, n_out=300)`: sort by temperature, split the range into
300 equal bins, take the **mean T** and **median mass** per bin. Median, not
mean, because medians ignore outlier spikes. 300 points is plenty for a curve
with a handful of steps and keeps each ODE solve in the fit fast.

Noise estimate: `1.4826 * median(|Δm - median(Δm)|) * sqrt(0.5)` over raw
point-to-point differences — the robust (MAD-based) standard deviation of the
measurement noise, corrected for differencing (√2). It becomes the starting
value of the noise parameter in the likelihood.

`smooth(y, w)`: plain centered moving average with edge padding. Used only
for derivative computations, never on the data that gets fitted.

### 2.3 Step detection (`detect_steps`)

1. Smooth the mass curve (window ≈ n/25), differentiate w.r.t. T with
   `np.gradient`, negate → DTG curve (mass-loss rate per kelvin), smooth
   again.
2. Peaks = local maxima above `0.08 * max(rate)`, merged if closer than 25 K.
   The 8% threshold is the detector's sensitivity knob for small peaks.
3. Shoulders = hidden components with no local maximum of their own (they
   ramp into a bigger peak). Signature: a significant dip in |d(rate)/dT| on
   a flank — the rate decelerates, then the main peak re-accelerates it. A
   candidate must sit at 12–80% of the maximum rate, ≥30 K from any peak,
   must not be a valley (rate below both flanks), and the slope must fall to
   under 1/1.6 of its maxima within 60 K on both sides. Detected shoulders
   are tagged `kind="shoulder"` and printed with a "(shoulder)" marker. This
   is what makes materials like heavily-filled silicones (broad shoulder +
   sharp peak) solvable without hand-editing the problem file.
4. Boundaries: between two peaks, the DTG minimum (original rule, unchanged —
   materials without shoulders are processed exactly as before). Between a
   shoulder and a peak there is usually no minimum, so the boundary is the
   steepest point of the rate between them (the peak's inflection, where the
   sharp component takes over).
5. Per step: Δm = smoothed-mass drop across its boundaries; FWHM = width where
   the DTG stays above half the feature value (shoulders use twice their left
   half-width, since their right side merges into the peak); peak steps with
   Δm < 0.005 are discarded, shoulder steps need Δm ≥ 0.03 — a shoulder must
   carry real mass to justify its three parameters.

### 2.4 Initial guesses (`arrhenius_guess`)

For a first-order reaction under a linear ramp, theory gives two relations
that we invert:

- Peak width:  `E ≈ 2.45·R·Tp²/FWHM`  (sharper peak ⇒ higher barrier).
  Clamped to [40, 400] kJ/mol to survive pathological widths.
- Peak position (rate maximal at Tp):  `A = (β·E/(R·Tp²))·exp(E/(R·Tp))`.
  Note β must be in K/s here.
- n starts at 1.

These are deliberately crude; they only need to land inside the optimizer's
basin of convergence, and empirically they always have.

### 2.5 Output writers (`emit`, `emit_joint`)

**`emit(path, out_root)`** — the per-file writer. Sequence: read → condition
→ detect → print a human-readable summary (this printout is your first
quality gate — always read it) → build the TOML line list → write the data
CSV, the steady-state CSV (if measured), the TOML, and a matplotlib
diagnostic PNG (curve + DTG + detected step markers).

Model bookkeeping worth understanding:
- `residual = max(m[-1], 0)` — a fully burnt sample can drift slightly
  negative (buoyancy/drift); mass can't be negative, so clamp with a printed
  note.
- Step sizes are rescaled by `(1 - residual)/Σdm` so that
  `ΣP_i + Residue = 1` exactly — mass balance by construction.
- Every reaction is written `aerobic = false` because single-atmosphere data
  cannot identify an oxygen order anyway (see §4.2).

**`emit_joint(paths, out_root, name)`** — same ingredients, one problem for
several runs of the same material. Design decisions:
- The **step structure comes from the run that resolves the most steps**
  (slow runs merge peaks; fast runs can separate them — with the fiber data
  the 10 °C/min run resolved a step the others hid).
- The **residue is the mean** of the runs' final plateaus, clamped ≥ 0, and a
  warning fires when the runs disagree by more than 2% — that either means
  specimen variability (AF340) or accidentally mixed materials.
- Each run keeps its own measured β and its own CSV; `[experiments]` lists
  them in parallel arrays. The kinetic parameters are shared — that's the
  whole point of a joint fit.

---

## 3. The problem file (TOML) — the contract between the halves

Everything the Julia side knows arrives through this file:

| Section | Meaning |
|---|---|
| `[grid] T0, T_final` | simulated temperature span [K], taken from the data |
| `[[species]]` | one entry per component: `initial` mass fraction, `gas = true` excludes it from the observable solid mass |
| `[[reaction]]` | `reactant`, `products`+`yields` (mass-balanced), nominal `log10_A`, `E`, `n`, optional `aerobic`+`m`, and `estimate` = list of parameter names the optimizer may vary |
| `[experiments]` | parallel arrays `ids`/`beta`/`PO2`/`files` — several rows = joint multi-condition fit |
| `[data]` | `mode` (`csv` = real data, `synthetic` = twin-experiment self-test), `noise_sigma`, CSV column names |
| `[estimation] n_multistarts` | optimizer restarts |

Anything can be edited by hand and re-fit without touching code. Warning:
re-running the prep on the same file **regenerates** the TOML and overwrites
manual edits.

---

## 4. The Julia engine (`src/engine/`)

### 4.1 `spec.jl` — problem representation and validation

- `SpeciesSpec`, `ReactionSpec`, `ProblemSpec` — plain structs mirroring the
  TOML.
- `load_problem(path)` — TOML → structs, with type coercions.
- `experiments(spec)` — named tuples per run; **converts β from K/min (user
  units) to K/s** (kinetics run in seconds).
- `availability(spec)` — for each reactant, the total amount that ever
  exists: its initial amount plus everything produced by upstream reactions,
  resolved recursively through the production graph with cycle detection.
  This is the denominator of the conversion, see §4.2. For a component that
  is only consumed it's just its initial value; for an intermediate (e.g. a
  char produced by one reaction and burned by another) it's
  `initial + yield·availability(parent)`.
- `validate_problem(spec)` — errors (block the run: unknown species,
  length mismatches, β ≤ 0 …) and warnings (advisory: mass balance ≠ 1,
  single heating rate ⇒ A–E correlation, single PO₂ ⇒ oxygen order
  unidentifiable, one species consumed by several reactions).

### 4.2 `build.jl` — from spec to ODE system

The heart. Species and parameters are created **dynamically by name**
(`only(@species $(nm)(t))`, `only(@parameters $(nm))`) inside loops — that is
what makes the engine generic; nothing is hard-coded per material.

For every reaction the rate is built as:

```julia
frac = max(species, 0) / (availability + EPS)          # = (1 − α)
rate = avail * 10^log10_A * exp(-E/(R*T)) * max(frac, POW_FLOOR)^n
if aerobic:  rate *= max(PO2/PO2_ref, POW_FLOOR)^m
```

Line-by-line reasoning:
- **Temperature is itself a state variable** with `dT/dt = β` (a zeroth-order
  "reaction"). This keeps everything inside one ODE system.
- **`(1−α)^n` instead of `[species]^n`:** conversion-based kinetics are the
  TGA standard; the order acts on the *normalized* remaining fraction, so n
  means the same thing regardless of how large the component is.
- **`avail *` out front:** converts dα/dt (conversion per second) into mass
  per second.
- **`max(species, 0)`:** the integrator can momentarily step a species a hair
  below zero; a negative base under a fractional power is NaN.
- **`POW_FLOOR = 1e-10` floor before `^n`:** subtle and critical. n is an
  *estimated* parameter, and ForwardDiff computes ∂(x^n)/∂n = x^n·log(x),
  which is NaN at x = 0. When a component burns out completely, x hits exactly
  0 and, without the floor, the whole gradient — and therefore the optimizer —
  silently dies. This bug produced "all starts failed" until diagnosed.
- **Oxygen factor `(PO2/PO2_ref)^m`:** normalized so it equals exactly 1 in
  air (0.21). Fitted kinetics from air data therefore don't depend on m at
  all; m only matters once experiments at other oxygen levels exist.
- **`ReactionSystem(rxns, t, sts, ps)` with explicit species/parameter
  lists:** Catalyst silently drops species that appear in no reaction — which
  is exactly what an inert residue is. Passing the lists explicitly keeps it.

Also here: `param_values(spec; override)` (name → numeric value for every
parameter, with fitted values overriding nominals) and `u0_map` (initial
conditions).

### 4.3 `data.jl` — simulation and measurements

- `simulate(h, spec, beta, PO2; override, n_points)` — builds the parameter
  map, computes `t_final = (T_final − T0)/β`, solves with **Rodas5P** (a
  stiff solver — Arrhenius nonlinearity makes these systems stiff) at tight
  tolerances (abstol 1e-10 / reltol 1e-8), and returns times, temperatures,
  every species trajectory, and the total *solid* mass (gas species
  excluded).
- `load_csv_data(spec)` — reads each experiment's CSV and reconstructs the
  time axis from temperature: `t = max((T − T0)/β, 0)`. The `max(…, 0)`
  clamp exists because binning jitter can put the first data point a hair
  below T0, and PEtab rejects negative measurement times.
- `generate_synthetic(h, spec)` — the twin-experiment mode: simulate with the
  nominal parameters, add Gaussian noise, return as measurements. Used by the
  self-test that proved the estimator recovers known parameters (~2%).
- `get_measurements` — dispatch on `spec.data_mode`.

### 4.4 `inverse.jl` — the estimation

- Bounds around nominals: log₁₀A ± 3, E ∈ [0.6v, 1.5v],
  n ∈ [max(0.1, v−1), v+1.5], m ∈ [0.05, v+1.2]. Purpose: keep the Arrhenius
  exponential from overflowing during the search. **A fitted value sitting on
  a bound is a diagnostic** — either widen (edit the nominal in the TOML and
  re-fit) or, if it re-saturates (fiber case), the model family itself is
  inadequate.
- `setup_petab(h, spec, measurements)` — the PEtab formulation:
  - observable: `mass_total = Σ solid species` with noise parameter
    `sigma_mass` (estimated, log scale, bounds 1e-4…0.1);
  - one `PEtabCondition` per experiment mapping that run's β and PO₂;
  - one `PEtabParameter` per estimated quantity, linear scale;
  - everything not estimated (yields, availabilities, non-estimated kinetics)
    goes into the fixed `parameter_map`;
  - `PEtabODEProblem` with `gradient_method = :ForwardDiff` — exact
    gradients through the ODE solve, which is what makes L-BFGS reliable
    here.
- `run_estimation(prob; n_multistarts)` — start 1 is the nominal guess;
  further starts perturb it by ±20% of each bound width (uniform), clamped.
  Deliberately **not** Latin-hypercube over the whole box: random Arrhenius
  combinations routinely overflow/stall the integrator, wasting most starts.
  Failed starts are caught and skipped; the best `fmin` wins. Fixed RNG seed
  → reproducible.
- `compare` / `print_comparison` — nominal vs fitted table. For real data the
  header says "Initial guess vs Fitted / % Moved" because there is no
  ground truth; only in synthetic mode is it a genuine "% Error".

### 4.5 `engine.jl` — the pipeline function

`solve_problem(spec)`: validate (print warnings; abort on errors) → build →
get measurements (+ overview plot) → setup PEtab → estimate → comparison
table + CSV → validation plot. Returns everything for interactive use.

### 4.6 `fit_report.jl` — metrics and figures

- `interp_linear` — model curve is dense in temperature; interpolate it onto
  the measured temperatures so residuals compare like with like.
- `fit_report(toml)` — loads the problem AND the fitted values from
  `results/<name>_parameter_comparison.csv` (the `override` dict), simulates,
  prints RMSE/MAE/max/R² per experiment, draws the two-panel
  data-vs-fit + residual figure. **Pitfall:** if you re-prep a problem so its
  step structure changes, an old CSV silently mismatches — re-fit before
  re-reporting.
- `paper_style_figure` — mass loss + per-component curves + DTG panel. The
  DTG of the *data* is a smoothed finite difference; of the model, a plain
  finite difference (it's already smooth). Both ×β to give 1/s units.
- `paper_style_multi` — joint problems: all runs overlaid, data and fit per
  color, DTG panel below; species curves omitted (they differ per rate).
- Conventions everywhere: computation in Kelvin, display in °C
  (`toC(x) = x .- 273.15`); model curves drawn **only over the measured
  range** (`keep = sim.T .<= max(data.T)`); measured steady-state points and
  "not in export" bands drawn when a `_steady_state.csv` exists.
- The autorun at the bottom is guarded by
  `abspath(PROGRAM_FILE) == @__FILE__` so the file can also be `include`d as
  a library (that's how `analyze.py`'s template uses it).

---

## 5. Design decisions FAQ (the "why" list)

- **Why fit the mass curve, not dα/dT?** Differentiating measured data
  amplifies noise; fitting the integral curve uses the raw measurement. The
  DTG is still shown in figures because it's the more sensitive *visual*
  check.
- **Why log₁₀A?** A spans 10¹–10²⁰ across problems; optimizers need
  comparable parameter scales.
- **Why Kelvin internally?** exp(−E/RT) is only meaningful on an absolute
  scale. °C is display-only.
- **Why multistart L-BFGS and not a global optimizer?** With physics-based
  starting values the basin is almost always right; multistart guards against
  the rest at a fraction of a global search's cost.
- **Why a joint fit at all?** One curve cannot pin A and E separately
  (compensation). Sharing parameters across heating rates is both the cure
  and a validity test: AF340 passed (transferable kinetics), the fiber data
  failed (transport-limited — no transferable triplet exists). Both outcomes
  are informative; see `results/fiber/fiber_cross_prediction.png` for the
  definitive illustration.
- **Why is the residue inert?** Because within the measured range it never
  reacted — the model claims nothing the data didn't show.
- **Why pseudo-components?** Mass measurement alone cannot identify
  chemistry; P1/P2 honestly represent "fractions with distinct kinetics".

## 6. Known pitfalls (read before modifying)

1. Re-running prep overwrites hand-edited TOMLs — a `.toml.bak` backup is
   saved automatically whenever the content differs, so edits are
   recoverable.
2. `fit_report` verifies that `results/<name>_parameter_comparison.csv`
   matches the problem's estimated parameters and refuses with a clear error
   if the problem changed since the fit — re-run `solve.jl` first.
3. A parameter at its bound = the optimizer wanted more room. Widen via the
   TOML nominal, or read it as model inadequacy if it saturates again.
4. .xls exports truncate at 65,536 rows — use .txt / .xlsx.
5. Time units differ between export flavors; the units row is authoritative
   (the reader handles s/min, warns when absent and implausible).
6. Julia buffers stdout when redirected to a file — logs appear only at
   process exit.
7. After any engine change, run `python analyze.py --selftest` — the
   synthetic round-trip in `tests/selftest.toml` must recover its known
   parameters (mean error < 10%; historically ~2%).
8. Detector sensitivity is adjustable: `emit(..., thr_frac=0.03)` or
   `python tools/prepare_tga.py file --sensitivity 0.03` resolves smaller
   DTG peaks (default 0.08). Shoulders (components with no local maximum)
   are detected automatically via the slope-dip criterion in §2.3 and need
   no tuning; their own guard is `shoulder_min_dm` (default 3% of mass).

## 7. How to make common changes

- **Rename a component** (nicer legends): edit `name`/`reactant` pairs in the
  TOML, re-run `solve.jl` + `fit_report.jl`.
- **Add/remove a step by hand:** add a `[[species]]` + `[[reaction]]` pair,
  adjust the initials so they sum to 1 with the residue.
- **Change detector sensitivity:** `thr = 0.08 * rate.max()` and
  `min_dm=0.005` in `detect_steps`.
- **Wider bounds for one parameter:** move its nominal in the TOML (bounds
  are derived from nominals in `inverse.jl:_b_*`).
- **A new rate-law family** (e.g. autocatalytic α^p·(1−α)^n): extend the rate
  expression in `build.jl` (add the `@parameters` and the factor), add the
  nominal+estimate plumbing in `spec.jl`/`inverse.jl`, and teach `emit` to
  write it. The dynamic-name pattern makes this a ~30-line change.
- **Different atmospheres:** set `aerobic = true` + `m` on oxidation steps
  and give experiments different `PO2` values — the machinery is already
  there.
