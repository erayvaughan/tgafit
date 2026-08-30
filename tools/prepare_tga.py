#!/usr/bin/env python
"""
Generic TGA-data preparation: instrument export -> engine-ready problem.

Takes ANY thermogravimetric export (TA TRIOS .xls/.xlsx, or plain .csv/.txt with
temperature + weight columns), and produces everything the Julia engine needs:

  1. data/<name>.csv          clean (temperature [K], mass [0-1]) curve
  2. problems/<name>.toml     auto-generated problem: one pseudo-component per
                              detected mass-loss step + inert residue, with
                              physics-based initial guesses for A, E, n
  3. results/<name>_prep.png  diagnostic plot (curve, DTG, detected steps)

Then fit with:   julia --project=. solve.jl problems/<name>.toml

Method notes
  - Steps are detected as peaks of the smoothed mass-loss rate (-dm/dT).
  - Initial guesses per step (refined later by the optimizer):
      E  from peak width:  E ~= 2.45*R*Tp^2 / FWHM   (1st-order approximation)
      A  from peak condition at linear heating:  A = (beta*E/(R*Tp^2))*exp(E/(R*Tp))
      n  = 1.0
  - The model is a parallel pseudo-component scheme (standard for TGA):
      P_i -> Gas   for each step,  plus an inert Residue.

Usage:
    python tools/prepare_tga.py <file1> [file2 ...]
"""
import os, re, sys
import numpy as np

R_GAS = 8.314

# ----------------------------------------------------------------------------
# 1. Reading: find the heating-ramp (temperature, weight) block in any export
# ----------------------------------------------------------------------------

def _numeric_block(df):
    """Locate header row with Temperature & Weight columns, return (T, W, t)."""
    import pandas as pd
    hdr_row, cols = None, None
    for i in range(min(len(df), 30)):
        row = [str(x).strip().lower() for x in df.iloc[i].tolist()]
        if any("temperature" in c for c in row) and any(c.startswith("weight") or "mass" in c for c in row):
            hdr_row = i
            cols = row
            break
    if hdr_row is None:
        return None
    def find(*keys):
        for j, c in enumerate(cols):
            if any(k in c for k in keys):
                return j
        return None
    jT = find("temperature")
    jW = find("weight", "mass")
    jt = find("time")
    # the row after the header often holds units (e.g. "s"/"min" for Time)
    time_unit = ""
    if jt is not None and hdr_row + 1 < len(df):
        u = str(df.iloc[hdr_row + 1, jt]).strip().lower()
        if u in ("s", "sec", "secs", "seconds", "min", "mins", "minute", "minutes"):
            time_unit = u
    block = df.iloc[hdr_row + 1:].apply(pd.to_numeric, errors="coerce")
    keep = block[jT].notna() & block[jW].notna()
    block = block[keep]
    if len(block) < 50:
        return None
    T = block[jT].to_numpy(float)
    W = block[jW].to_numpy(float)
    t = block[jt].to_numpy(float) if jt is not None else None
    return T, W, t, time_unit

def _read_lims_txt(path):
    """
    Parse a TA Instruments LIMS text export (.txt): [Step] blocks with a
    tab-separated 'Variables' header and 'Data point' rows. Returns candidates
    in the same (name, (T, W, t_seconds), nrows) form as the Excel reader.
    """
    candidates = []
    step_name, names, units, jT = None, None, None, None
    jW = jt = None
    T, W, t = [], [], []

    def flush():
        if names is not None and len(T) > 50:
            tt = np.array(t, float)
            if units is not None and jt is not None and "min" in units[jt].lower():
                tt = tt * 60.0                      # LIMS time is in minutes
            candidates.append((step_name or "step",
                               (np.array(T, float), np.array(W, float), tt, "s"),
                               len(T)))

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("[Step]"):
                flush()
                step_name, names, units, jT, jW, jt = None, None, None, None, None, None
                T, W, t = [], [], []
            elif line.startswith("Step name\t"):
                step_name = line.split("\t", 1)[1]
            elif line.startswith("Variables\t"):
                names = [c.strip().lower() for c in line.split("\t")[1:]]
                jT = next((j for j, c in enumerate(names) if "temperature" == c), None)
                if jT is None:
                    jT = next((j for j, c in enumerate(names) if "temperature" in c), None)
                jW = next((j for j, c in enumerate(names) if c.startswith("weight")), None)
                jt = next((j for j, c in enumerate(names) if "time" == c), 0)
            elif line.startswith("Units\t"):
                units = line.split("\t")[1:]
            elif line.startswith("Data point\t") and jT is not None and jW is not None:
                v = line.split("\t")[1:]
                try:
                    T.append(float(v[jT])); W.append(float(v[jW])); t.append(float(v[jt]))
                except (ValueError, IndexError):
                    pass
    flush()
    return candidates

def _is_lims(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(200)
        return "[Version]" in head and "LIMS export" in head
    except OSError:
        return False

def read_export(path):
    """Return dict(T_K, mass_frac, beta_Kmin, truncated) for the heating ramp."""
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    candidates = []
    if ext in (".txt", ".csv") and _is_lims(path):
        candidates = _read_lims_txt(path)
    elif ext in (".xls", ".xlsx"):
        xl = pd.ExcelFile(path)
        for sh in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=sh, header=None)
            blk = _numeric_block(df)
            if blk is not None:
                candidates.append((sh, blk, len(df)))
    else:
        for sep in [",", ";", "\t", None]:
            try:
                df = pd.read_csv(path, sep=sep, header=None, engine="python")
                blk = _numeric_block(df)
                if blk is not None:
                    candidates.append(("csv", blk, len(df)))
                    break
            except Exception:
                continue
    if not candidates:
        raise SystemExit(f"could not find a (Temperature, Weight) data block in {path}")

    # pick the heating segment: largest temperature RISE
    best, rise = None, -1
    for sh, (T, W, t, tunit), nrows in candidates:
        r = T[-1] - T[0]
        if r > rise:
            rise, best = r, (sh, T, W, t, tunit, nrows)
    sh, T, W, t, tunit, nrows = best
    if rise < 50:
        raise SystemExit(f"no heating ramp found in {path} (max temperature rise {rise:.0f})")

    # units: C -> K if values look like Celsius
    T_K = T + 273.15 if T.max() < 1500 and T.min() < 200 else T.copy()
    mass = W / W[0]                       # normalize (works for % or mg)

    # time to seconds, using the export's units row
    if t is not None and tunit.startswith("min"):
        t = t * 60.0

    beta = None
    if t is not None and t[-1] > t[0]:
        beta = float(np.polyfit(t, T, 1)[0] * 60.0)  # K/min
        if beta > 150 and not tunit:
            # no units row and implausibly fast ramp: the time column was almost
            # certainly minutes. Flag it loudly rather than fit garbage.
            print(f"  [!] computed heating rate {beta:.0f} K/min is implausible and the file "
                  f"has no time-units row; assuming time in minutes -> {beta/60:.2f} K/min")
            beta /= 60.0
    if beta is None or beta <= 0:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:c|k)\s*(?:min|/min|min-1)", os.path.basename(path).lower())
        beta = float(m.group(1)) if m else 10.0

    truncated = nrows >= 65530 and ext == ".xls"   # legacy .xls row-limit hit

    # Measured steady state: if a COOLING segment exists that starts above the
    # heating data's end (e.g. TRIOS cool-down sheet), its first points are a
    # raw measurement of the final state — the sample mass at max temperature.
    is_celsius = T.max() < 1500 and T.min() < 200
    steady = None
    for sh2, (T2, W2, t2, _u2), _n in candidates:
        if T2[0] - T2[-1] > 50 and T2.max() > T.max() + 10:   # cooling, beyond heating end
            sel = T2 >= (T2.max() - 3.0)
            if sel.sum() >= 3:
                sT = float(np.mean(T2[sel])) + (273.15 if is_celsius else 0.0)
                sW = float(np.median(W2[sel])) / W[0]
                steady = (sT, sW)
            break

    return dict(sheet=sh, T=T_K, mass=mass, beta=beta, truncated=truncated, steady=steady)

# ----------------------------------------------------------------------------
# 2. Condition: downsample on a uniform temperature grid + noise estimate
# ----------------------------------------------------------------------------

def condition(T, mass, n_out=300):
    order = np.argsort(T)
    T, mass = T[order], mass[order]
    grid = np.linspace(T[0], T[-1], n_out + 1)
    Tc, Mc = [], []
    for i in range(n_out):
        sel = (T >= grid[i]) & (T < grid[i + 1])
        if sel.sum() > 0:
            Tc.append(T[sel].mean())
            Mc.append(np.median(mass[sel]))
    Tc, Mc = np.array(Tc), np.array(Mc)
    # noise: robust std of point-to-point residual on the raw curve
    d = np.diff(mass)
    sigma = float(max(1.4826 * np.median(np.abs(d - np.median(d))) * np.sqrt(0.5), 1e-4))
    return Tc, Mc, sigma

def smooth(y, w):
    w = max(3, int(w) | 1)
    k = np.ones(w) / w
    ypad = np.r_[np.full(w // 2, y[0]), y, np.full(w // 2, y[-1])]
    return np.convolve(ypad, k, mode="valid")

# ----------------------------------------------------------------------------
# 3. Step detection on the DTG curve (-dm/dT)
# ----------------------------------------------------------------------------

def detect_steps(T, m, min_dm=0.005, thr_frac=0.08, shoulder_min_dm=0.03,
                 sh_rate_lo=0.12, sh_rate_hi=0.80, sh_dist=30.0):
    """Find decomposition steps as DTG peaks PLUS shoulders.

    thr_frac: peak threshold as a fraction of the largest DTG peak. Lower it
    (e.g. 0.03) to resolve small peaks; raise it to ignore them.

    Shoulders: a hidden component that ramps into a bigger peak produces no
    local maximum of the rate, only a deceleration followed by re-acceleration,
    i.e. a significant dip in |d(rate)/dT| on a flank. Those are detected
    separately and marked kind="shoulder". Peak-to-peak boundaries keep the
    original valley-minimum rule, so materials without shoulders are processed
    exactly as before.
    """
    ms = smooth(m, len(m) // 25)
    rate = -np.gradient(ms, T)                     # mass loss per K
    rate = smooth(rate, len(m) // 30)
    thr = thr_frac * rate.max()

    # --- peaks: local maxima above threshold, separated by >= 25 K ---
    peaks = []
    for i in range(2, len(rate) - 2):
        if rate[i] >= rate[i-1] and rate[i] >= rate[i+1] and rate[i] > thr:
            if peaks and T[i] - T[peaks[-1]] < 25.0:
                if rate[i] > rate[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    if not peaks:
        return [], rate

    # --- shoulders: significant local minima of |slope| on a flank ---
    rn = rate / rate.max()
    slope = smooth(np.gradient(rn, T), len(T) // 30)
    asl = np.abs(slope)
    n = len(T)
    shoulders = []
    for i in range(3, n - 3):
        if not (asl[i] <= asl[i-1] and asl[i] <= asl[i+1]):
            continue
        if not (sh_rate_lo < rn[i] < sh_rate_hi):
            continue
        if any(abs(T[i] - T[p]) < sh_dist for p in peaks):
            continue
        lo = np.argmin(np.abs(T - (T[i] - 25))); hi = np.argmin(np.abs(T - (T[i] + 25)))
        if rate[i] < rate[lo] and rate[i] < rate[hi]:
            continue                                    # valley, not a shoulder
        wl = (T >= T[i] - 60) & (T < T[i]); wr = (T > T[i]) & (T <= T[i] + 60)
        if not wl.any() or not wr.any():
            continue
        ref = asl[i] + 1e-5
        if asl[wl].max() >= 1.6 * ref and asl[wr].max() >= 1.6 * ref:
            shoulders.append(i)
    merged = []
    for i in shoulders:
        if merged and T[i] - T[merged[-1]] < 40:
            if asl[i] < asl[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    shoulders = merged

    # --- boundaries between features, then per-step quantities ---
    feats = sorted([(p, "peak") for p in peaks] + [(sh, "shoulder") for sh in shoulders])

    def boundary(a, b, kind_a, kind_b):
        seg = rate[a:b]
        j = int(np.argmin(seg))
        if kind_a == "peak" and kind_b == "peak":
            return a + j                            # original rule, unchanged
        if seg[j] < 0.9 * min(rate[a], rate[b]):
            return a + j                            # genuine valley
        if b - a < 3:
            return (a + b) // 2
        # shoulder handover: steepest point of the rate between the features
        return a + int(np.argmax(np.abs(np.gradient(seg))))

    while True:
        bounds = [0]
        for (fa, ka), (fb, kb) in zip(feats[:-1], feats[1:]):
            bounds.append(boundary(fa, fb, ka, kb))
        bounds.append(n - 1)
        steps, drop = [], None
        for k, (f, kind) in enumerate(feats):
            i0, i1 = bounds[k], bounds[k+1]
            dm = float(ms[i0] - ms[i1])
            lim = shoulder_min_dm if kind == "shoulder" else min_dm
            if dm < lim:
                if kind == "shoulder":
                    drop = k
                    break
                continue
            half = rate[f] / 2
            iL = f
            while iL > i0 and rate[iL] > half: iL -= 1
            if kind == "peak":
                iR = f
                while iR < i1 and rate[iR] > half: iR += 1
                fwhm = max(float(T[iR] - T[iL]), 5.0)
            else:
                fwhm = max(2.0 * float(T[f] - T[iL]), 20.0)
            steps.append(dict(Tp=float(T[f]), fwhm=fwhm, dm=dm, i0=i0, i1=i1, kind=kind))
        if drop is None:
            return steps, rate
        feats.pop(drop)

def arrhenius_guess(step, beta_Kmin):
    """Initial guesses from peak temperature + width (1st-order theory)."""
    Tp, fwhm = step["Tp"], step["fwhm"]
    E = 2.45 * R_GAS * Tp**2 / fwhm
    E = float(np.clip(E, 4.0e4, 4.0e5))
    beta_Ks = beta_Kmin / 60.0
    A = (beta_Ks * E / (R_GAS * Tp**2)) * np.exp(E / (R_GAS * Tp))
    return float(np.log10(max(A, 1e-2))), E

# ----------------------------------------------------------------------------
# 4. Emit CSV + problem TOML
# ----------------------------------------------------------------------------

def emit(path, out_root, thr_frac=0.08):
    name = re.sub(r"[^a-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0].lower()).strip("_")
    data_dir = os.path.join(out_root, "data")
    prob_dir = os.path.join(out_root, "problems")
    res_dir  = os.path.join(out_root, "results")
    for d in (data_dir, prob_dir, res_dir):
        os.makedirs(d, exist_ok=True)

    exp = read_export(path)
    T, m, sigma = condition(exp["T"], exp["mass"])
    steps, rate = detect_steps(T, m, thr_frac=thr_frac)
    if not steps:
        raise SystemExit(f"{path}: no mass-loss steps detected")

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  sheet: {exp['sheet']}   beta: {exp['beta']:.3f} K/min   noise sigma ~ {sigma:.4f}")
    print(f"  range: {T[0]:.0f}..{T[-1]:.0f} K   final residual: {m[-1]:.4f}")
    if exp["truncated"]:
        print("  [!] legacy .xls row limit hit — data is TRUNCATED before the programmed end")
        print("    (re-export as .xlsx/.csv for the full range; the pipeline handles it as-is)")

    residual = float(m[-1])
    if residual < 0:
        # full burnout with slight instrument drift below zero: mass cannot be
        # negative, so the physical residual is 0
        print(f"  note: final signal {residual:.4f} < 0 (drift after full burnout); residual set to 0")
        residual = 0.0
    dm_total = sum(s["dm"] for s in steps)
    scale = (1.0 - residual) / dm_total if dm_total > 0 else 1.0

    lines = [f"# Auto-generated by tools/prepare_tga.py from {os.path.basename(path)}",
             f"# Parallel pseudo-component TGA model: one reaction per detected mass-loss step.",
             f"# Initial A/E guesses from DTG peak position/width; refined by the optimizer.",
             f'name = "{name}"', "",
             "[grid]",
             f"T0      = {T[0]:.2f}",
             f"T_final = {T[-1]:.2f}",
             "R_gas   = 8.314",
             "PO2_ref = 0.21", ""]

    for k, s in enumerate(steps, 1):
        w = s["dm"] * scale
        lA, E = arrhenius_guess(s, exp["beta"])
        tag = "  (shoulder)" if s.get("kind") == "shoulder" else ""
        print(f"  step {k}: Tpeak={s['Tp']:.0f} K  FWHM={s['fwhm']:.0f} K  dm={w:.4f}"
              f"  ->  log10A={lA:.2f}  E={E/1000:.0f} kJ/mol{tag}")
        lines += [f"[[species]]", f'name = "P{k}"     # pseudo-component, DTG {s.get("kind","peak")} {s["Tp"]:.0f} K',
                  f"initial = {w:.5f}", "gas = false", ""]
    lines += ["[[species]]", 'name = "Residue"   # inert (final residual mass)',
              f"initial = {residual:.5f}", "gas = false", "",
              "[[species]]", 'name = "G"         # evolved gas (lumped)',
              "initial = 0.0", "gas = true", ""]

    for k, s in enumerate(steps, 1):
        lA, E = arrhenius_guess(s, exp["beta"])
        lines += ["[[reaction]]",
                  f'name = "step{k}"',
                  f'reactant = "P{k}"',
                  'products = ["G"]',
                  "yields   = [1.0]",
                  "aerobic  = false   # single-atmosphere data: O2 order not identifiable",
                  f"log10_A  = {lA:.4f}",
                  f"E        = {E:.1f}",
                  "n        = 1.0",
                  'estimate = ["log10_A", "E", "n"]', ""]

    csv_rel = f"data/{name}.csv"
    lines += ["[experiments]",
              f'ids  = ["{name}"]',
              f"beta = [{exp['beta']:.4f}]",
              "PO2  = [0.21]",
              f'files = ["{csv_rel}"]', "",
              "[data]",
              'mode = "csv"',
              f"noise_sigma = {sigma:.5f}",
              "n_points = 300",
              "seed = 42",
              'temp_col = "temperature"',
              'mass_col = "mass"', "",
              "[estimation]",
              "n_multistarts = 3", ""]

    # measured steady state from the cooling segment (raw data, shown on plots;
    # NOT used in the fit — the heating path to it is absent from the export)
    if exp.get("steady"):
        sT, sW = exp["steady"]
        ss_path = os.path.join(data_dir, f"{name}_steady_state.csv")
        with open(ss_path, "w") as fh:
            fh.write("temperature,mass\n")
            fh.write(f"{sT:.2f},{sW:.5f}\n")
        gap = m[-1] - sW
        print(f"  measured steady state (cooling-segment start): W={sW:.4f} at {sT-273.15:.0f} C")
        print(f"    -> unexported mass loss between {T[-1]-273.15:.0f} C and there: {gap:.4f}")
        print(f"  -> {ss_path}")

    csv_path = os.path.join(data_dir, f"{name}.csv")
    with open(csv_path, "w") as f:
        f.write("temperature,mass\n")
        for a, b in zip(T, m):
            f.write(f"{a:.3f},{b:.5f}\n")
    toml_path = os.path.join(prob_dir, f"{name}.toml")
    new_content = "\n".join(lines)
    # protect hand edits: if an existing TOML differs from what we are about to
    # write, keep a .bak copy so renames/tweaks are recoverable
    if os.path.isfile(toml_path):
        with open(toml_path) as f:
            old_content = f.read()
        if old_content.strip() != new_content.strip():
            bak = toml_path + ".bak"
            with open(bak, "w") as f:
                f.write(old_content)
            print(f"  note: existing {os.path.basename(toml_path)} differed — saved backup to {os.path.basename(bak)}")
    with open(toml_path, "w") as f:
        f.write(new_content)
    print(f"  -> {csv_path}  ({len(T)} points)")
    print(f"  -> {toml_path}  ({len(steps)} reaction step(s))")

    # diagnostic plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(T, m, "k-", lw=2, label="data (conditioned)")
    for k, s in enumerate(steps, 1):
        a1.axvline(s["Tp"], color=f"C{k}", ls="--", alpha=0.7)
        a1.text(s["Tp"], m.max(), f" step {k}", color=f"C{k}", rotation=90, va="top", fontsize=8)
    a1.set_ylabel("mass fraction"); a1.legend(); a1.grid(alpha=0.3)
    a1.set_title(f"{name}  ({exp['beta']:.1f} K/min)" + ("  [TRUNCATED .xls]" if exp["truncated"] else ""))
    a2.plot(T, rate, "b-", lw=1.5)
    a2.set_xlabel("Temperature [K]"); a2.set_ylabel("-dm/dT [1/K]"); a2.grid(alpha=0.3)
    png = os.path.join(res_dir, f"{name}_prep.png")
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"  -> {png}")
    return toml_path

def _stem(path):
    return re.sub(r"[^a-z0-9]+", "_",
                  os.path.splitext(os.path.basename(path))[0].lower()).strip("_")

def emit_joint(paths, out_root, name, thr_frac=0.08):
    """
    Build ONE multi-condition problem from several runs of the SAME material
    (different heating rates and/or atmospheres). Shared species/reactions;
    one [experiments] entry per file. Step detection uses the file where the
    most steps are resolved; initial fractions/residue are averaged over files.
    """
    data_dir = os.path.join(out_root, "data")
    prob_dir = os.path.join(out_root, "problems")
    os.makedirs(data_dir, exist_ok=True); os.makedirs(prob_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"JOINT PROBLEM '{name}' from {len(paths)} runs")
    print("=" * 60)
    exps = []
    for p in paths:
        exp = read_export(p)
        T, m, sigma = condition(exp["T"], exp["mass"])
        steps, _ = detect_steps(T, m, thr_frac=thr_frac)
        stem = _stem(p)
        with open(os.path.join(data_dir, f"{stem}.csv"), "w") as fh:
            fh.write("temperature,mass\n")
            for a, b in zip(T, m):
                fh.write(f"{a:.3f},{b:.5f}\n")
        print(f"  {stem}: beta={exp['beta']:.3f} K/min  range {T[0]:.0f}..{T[-1]:.0f} K  "
              f"residual {m[-1]:.4f}  steps {len(steps)}"
              + ("  [TRUNCATED .xls]" if exp["truncated"] else ""))
        exps.append(dict(stem=stem, beta=exp["beta"], T=T, m=m, sigma=sigma, steps=steps))

    ref = max(exps, key=lambda e: len(e["steps"]))
    if not ref["steps"]:
        raise SystemExit("no mass-loss steps detected in any run; cannot build a joint problem")
    print(f"  reference for step structure: {ref['stem']} ({len(ref['steps'])} steps)")
    residual = max(float(np.mean([e["m"][-1] for e in exps])), 0.0)
    resids = [float(e["m"][-1]) for e in exps]
    if max(resids) - min(resids) > 0.02:
        print(f"  [!] runs' final residuals differ by {max(resids)-min(resids):.3f} — "
              f"check that all files are the same material/specimen type")
    dm_total = sum(s["dm"] for s in ref["steps"])
    scale = (1.0 - residual) / dm_total if dm_total > 0 else 1.0
    T0 = min(float(e["T"][0]) for e in exps)
    Tf = max(float(e["T"][-1]) for e in exps)
    noise = max(e["sigma"] for e in exps)

    lines = [f"# Auto-generated JOINT problem: {len(exps)} runs of the same material",
             f"# fit simultaneously with shared kinetics (PEtab multi-condition).",
             f'name = "{name}"', "",
             "[grid]",
             f"T0      = {T0:.2f}", f"T_final = {Tf:.2f}",
             "R_gas   = 8.314", "PO2_ref = 0.21", ""]
    for k, s in enumerate(ref["steps"], 1):
        lines += ["[[species]]",
                  f'name = "P{k}"     # pseudo-component, DTG peak {s["Tp"]:.0f} K (ref run)',
                  f"initial = {s['dm']*scale:.5f}", "gas = false", ""]
    lines += ["[[species]]", 'name = "Residue"   # inert (mean final residual over runs)',
              f"initial = {residual:.5f}", "gas = false", "",
              "[[species]]", 'name = "G"', "initial = 0.0", "gas = true", ""]
    for k, s in enumerate(ref["steps"], 1):
        lA, E = arrhenius_guess(s, ref["beta"])
        lines += ["[[reaction]]", f'name = "step{k}"', f'reactant = "P{k}"',
                  'products = ["G"]', "yields   = [1.0]",
                  "aerobic  = false   # set true + vary PO2 when atmospheres differ",
                  f"log10_A  = {lA:.4f}", f"E        = {E:.1f}", "n        = 1.0",
                  'estimate = ["log10_A", "E", "n"]', ""]
    ids   = ", ".join(f'"{e["stem"]}"' for e in exps)
    betas = ", ".join(f"{e['beta']:.4f}" for e in exps)
    po2s  = ", ".join("0.21" for _ in exps)
    files = ", ".join(f'"data/{e["stem"]}.csv"' for e in exps)
    lines += ["[experiments]", f"ids  = [{ids}]", f"beta = [{betas}]",
              f"PO2  = [{po2s}]", f"files = [{files}]", "",
              "[data]", 'mode = "csv"', f"noise_sigma = {noise:.5f}",
              "n_points = 300", "seed = 42",
              'temp_col = "temperature"', 'mass_col = "mass"', "",
              "[estimation]", "n_multistarts = 2", ""]

    toml_path = os.path.join(prob_dir, f"{name}.toml")
    with open(toml_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> {toml_path}  ({len(ref['steps'])} shared steps × {len(exps)} experiments)")
    return toml_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tgafit_main")
    args = sys.argv[1:]
    thr = 0.08
    if "--sensitivity" in args:
        i = args.index("--sensitivity")
        thr = float(args[i + 1]); del args[i:i + 2]
    for p in args:
        emit(p, root, thr_frac=thr)
