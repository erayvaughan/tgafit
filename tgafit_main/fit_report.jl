# =============================================================================
# Fit report: quantitative data-vs-fitted-model comparison for ANY problem.
#
# Loads a problem TOML + the recovered parameters written by solve_problem
# (results/<name>_parameter_comparison.csv), simulates the fitted model, and
# reports RMSE / MAE / max error / R² against the experiment data, with a
# two-panel overlay + residual figure — same output as the paper comparison.
#
#   julia --project=. fit_report.jl problems/af512blanket_air_5_cmin.toml
# =============================================================================

include("src/engine/engine.jl")
using CSV, DataFrames, Plots, Printf, Statistics

function interp_linear(xs, ys, xq)
    out = similar(xq, Float64); n = length(xs)
    for (k, x) in enumerate(xq)
        if x <= xs[1];      out[k] = ys[1]
        elseif x >= xs[n];  out[k] = ys[n]
        else
            j = searchsortedlast(xs, x)
            t = (x - xs[j]) / (xs[j+1] - xs[j])
            out[k] = ys[j]*(1-t) + ys[j+1]*t
        end
    end
    out
end

function fit_report(cfgpath::AbstractString)
    spec = load_problem(cfgpath)
    cmp_path = "results/$(spec.name)_parameter_comparison.csv"
    isfile(cmp_path) || error("missing $cmp_path — run solve_problem first")
    cmp = CSV.read(cmp_path, DataFrame)
    # guard against stale results: the fitted-parameter file must match the
    # problem's current estimated parameters exactly, else the figures would
    # silently mix old fitted values with new nominals
    want = Set(string.(estimated_names(spec)))
    have = Set(String.(cmp.Parameter))
    if want != have
        error("fitted parameters in $cmp_path do not match problem '$(spec.name)' " *
              "(expected $(sort(collect(want))), found $(sort(collect(have)))). " *
              "The problem definition changed since the fit — re-run solve.jl first.")
    end
    override = Dict(Symbol(r.Parameter) => Float64(r.Recovered) for r in eachrow(cmp))

    h = build_system(spec)
    _, clean = load_csv_data(spec)

    toC(x) = x .- 273.15   # display in °C (kinetics stay in K internally)
    # measured steady state (from the export's cooling segment), if present
    ss_path = "data/$(spec.name)_steady_state.csv"
    ss = isfile(ss_path) ? CSV.read(ss_path, DataFrame) : nothing
    Tmax = ss === nothing ? spec.T_final : max(spec.T_final, maximum(ss.temperature) + 20)
    xlC = (spec.T0 - 273.15, Tmax - 273.15)
    p1 = plot(ylabel = "Mass fraction", xlims = xlC,
              title = "$(spec.name): data vs fitted model",
              legend = :bottomleft, grid = true, framestyle = :box)
    p2 = plot(xlabel = "Temperature [°C]", ylabel = "Model − data",
              xlims = xlC, legend = false, grid = true, framestyle = :box)
    hline!(p2, [0.0], color = :gray, linestyle = :dash)

    println("="^64)
    println("  Fit report — $(spec.name)")
    println("="^64)
    for (i, e) in enumerate(experiments(spec))
        d = clean[e.id]
        sim = simulate(h, spec, e.beta, e.PO2; override = override, n_points = 500)
        Mm = interp_linear(sim.temperatures, sim.mass, collect(d.temperatures))
        res = Mm .- d.mass_noisy
        rmse = sqrt(mean(res.^2)); mae = mean(abs.(res)); mx = maximum(abs.(res))
        r2 = 1 - sum(res.^2) / sum((d.mass_noisy .- mean(d.mass_noisy)).^2)
        @printf("  %-24s  RMSE %.4f   MAE %.4f   max %.4f   R² %.4f\n",
                String(e.id), rmse, mae, mx, r2)
        c = i == 1 ? :red : :blue
        scatter!(p1, toC(d.temperatures), d.mass_noisy, label = "$(e.id) data",
                 color = c, markersize = 2.5, markerstrokewidth = 0, alpha = 0.5)
        # model line only over the measured range — nothing drawn where data is absent
        keep = sim.temperatures .<= maximum(d.temperatures)
        plot!(p1, toC(sim.temperatures[keep]), sim.mass[keep], label = "fitted model",
              color = :black, linewidth = 2.5)
        plot!(p2, toC(d.temperatures), res, color = :purple, linewidth = 1.6)
        if ss !== nothing && i == 1
            vspan!(p1, [toC(maximum(d.temperatures)), toC(minimum(ss.temperature))],
                   color = :gray, alpha = 0.12, label = "not in export")
            scatter!(p1, toC(ss.temperature), ss.mass, label = "measured steady state",
                     marker = :star5, markersize = 9, color = :red)
        end
    end
    println("="^64)
    println("  Fitted parameters:")
    for r in eachrow(cmp)
        @printf("    %-24s  start %12.4g   fitted %12.4g\n", r.Parameter, r.True_Value, r.Recovered)
    end
    println("="^64)

    p = plot(p1, p2, layout = grid(2, 1, heights = [0.7, 0.3]), size = (820, 640), dpi = 150)
    out = "results/$(spec.name)_fit_report.png"
    savefig(p, out)
    println("  ✓ Figure saved to: $out")

    paper_style_figure(h, spec, override, clean)
end

"""Centered moving average (for the data DTG)."""
function _smooth(y::AbstractVector, w::Int)
    w = max(3, w | 1); half = w ÷ 2; n = length(y)
    [mean(@view y[max(1,i-half):min(n,i+half)]) for i in 1:n]
end

"""
    paper_style_figure(h, spec, override, clean)

ONERA-paper-style figure (cf. Dellinger et al. Fig. 2):
  (a) mass loss — data squares, fitted total, per-species curves
  (b) mass-loss rate d(m/m₀)/dt [1/s] — data vs fitted model
"""
function paper_style_figure(h, spec::ProblemSpec, override, clean)
    exps = experiments(spec)
    length(exps) > 1 && return paper_style_multi(h, spec, override, clean)
    e = exps[1]
    d = clean[e.id]
    sim = simulate(h, spec, e.beta, e.PO2; override = override, n_points = 600)

    toC(x) = x .- 273.15   # display in °C (kinetics stay in K internally)
    ss_path = "data/$(spec.name)_steady_state.csv"
    ss = isfile(ss_path) ? CSV.read(ss_path, DataFrame) : nothing
    Tmax = ss === nothing ? spec.T_final : max(spec.T_final, maximum(ss.temperature) + 20)
    xlC = (spec.T0 - 273.15, Tmax - 273.15)
    keep = sim.temperatures .<= maximum(d.temperatures)   # draw model on data range only

    # --- (a) mass loss + species ---
    pa = plot(ylabel = "m/m₀  [–]", xlims = xlC,
              legend = :bottomleft, grid = true, framestyle = :box,
              title = "$(spec.name) — $(round(e.beta*60, sigdigits=3)) °C/min")
    scatter!(pa, toC(d.temperatures), d.mass_noisy, label = "TGA data",
             marker = :square, markersize = 2.5, markerstrokewidth = 0.4,
             markerstrokecolor = :black, color = :white)
    plot!(pa, toC(sim.temperatures[keep]), sim.mass[keep], label = "Material (fit)",
          color = :black, linewidth = 2.5)
    styles = [:dash, :dot, :dashdot, :dashdotdot]
    for (i, s) in enumerate(solid_species(spec))
        plot!(pa, toC(sim.temperatures[keep]), sim.species[s.name][keep], label = String(s.name),
              color = :black, linewidth = 1.6, linestyle = styles[mod1(i, length(styles))],
              alpha = 0.8)
    end
    if ss !== nothing
        vspan!(pa, [toC(maximum(d.temperatures)), toC(minimum(ss.temperature))],
               color = :gray, alpha = 0.12, label = "not in export")
        scatter!(pa, toC(ss.temperature), ss.mass, label = "measured steady state",
                 marker = :star5, markersize = 9, color = :red)
    end

    # --- (b) mass-loss rate [1/s] ---
    # data: smoothed finite-difference dm/dT × β ;  model: gradient of sim
    Td = collect(d.temperatures); Md = _smooth(collect(d.mass_noisy), length(d.mass_noisy) ÷ 25)
    dmdT_d = [-(Md[min(i+1,end)] - Md[max(i-1,1)]) / (Td[min(i+1,end)] - Td[max(i-1,1)]) for i in eachindex(Td)]
    Ts = sim.temperatures[keep]; Ms = sim.mass[keep]
    dmdT_m = [-(Ms[min(i+1,end)] - Ms[max(i-1,1)]) / (Ts[min(i+1,end)] - Ts[max(i-1,1)]) for i in eachindex(Ts)]
    pb = plot(xlabel = "T  [°C]", ylabel = "d(m/m₀)/dt  [1/s]",
              xlims = xlC, legend = :topleft, grid = true, framestyle = :box)
    scatter!(pb, toC(Td), dmdT_d .* e.beta, label = "TGA data",
             marker = :square, markersize = 2.5, markerstrokewidth = 0.4,
             markerstrokecolor = :black, color = :white)
    plot!(pb, toC(Ts), dmdT_m .* e.beta, label = "Material (fit)", color = :black, linewidth = 2)

    p = plot(pa, pb, layout = grid(2, 1), size = (760, 860), dpi = 150,
             left_margin = 5Plots.mm)
    out = "results/$(spec.name)_paper_style.png"
    savefig(p, out)
    println("  ✓ Paper-style figure saved to: $out")
end

"""
Paper-style figure for JOINT (multi-experiment) problems: every run's data and
fitted curve overlaid (cf. the ONERA paper's multi-heating-rate Fig. 4), with a
DTG panel per run. Species breakdown is omitted — it differs per heating rate.
"""
function paper_style_multi(h, spec::ProblemSpec, override, clean)
    toC(x) = x .- 273.15
    xlC = (spec.T0 - 273.15, spec.T_final - 273.15)
    colors = [:blue, :red, :green, :orange, :purple]

    pa = plot(ylabel = "m/m₀  [–]", xlims = xlC, legend = :bottomleft,
              grid = true, framestyle = :box,
              title = "$(spec.name) — joint fit, $(length(experiments(spec))) runs")
    pb = plot(xlabel = "T  [°C]", ylabel = "d(m/m₀)/dt  [1/s]", xlims = xlC,
              legend = :topleft, grid = true, framestyle = :box)

    for (i, e) in enumerate(experiments(spec))
        d = clean[e.id]; c = colors[mod1(i, length(colors))]
        lab = "$(round(e.beta*60, sigdigits=3)) °C/min"
        sim = simulate(h, spec, e.beta, e.PO2; override = override, n_points = 500)
        keep = sim.temperatures .<= maximum(d.temperatures)
        scatter!(pa, toC(d.temperatures), d.mass_noisy, label = lab * " data",
                 marker = :square, markersize = 2.2, markerstrokewidth = 0.3,
                 markerstrokecolor = c, color = :white)
        plot!(pa, toC(sim.temperatures[keep]), sim.mass[keep], label = lab * " fit",
              color = c, linewidth = 2.2)

        Td = collect(d.temperatures)
        Md = _smooth(collect(d.mass_noisy), length(d.mass_noisy) ÷ 25)
        dmdT_d = [-(Md[min(j+1,end)] - Md[max(j-1,1)]) / (Td[min(j+1,end)] - Td[max(j-1,1)]) for j in eachindex(Td)]
        Ts = sim.temperatures[keep]; Ms = sim.mass[keep]
        dmdT_m = [-(Ms[min(j+1,end)] - Ms[max(j-1,1)]) / (Ts[min(j+1,end)] - Ts[max(j-1,1)]) for j in eachindex(Ts)]
        scatter!(pb, toC(Td), dmdT_d .* e.beta, label = "",
                 marker = :square, markersize = 2.0, markerstrokewidth = 0.3,
                 markerstrokecolor = c, color = :white)
        plot!(pb, toC(Ts), dmdT_m .* e.beta, label = lab, color = c, linewidth = 1.8)
    end

    p = plot(pa, pb, layout = grid(2, 1), size = (760, 860), dpi = 150,
             left_margin = 5Plots.mm)
    out = "results/$(spec.name)_paper_style.png"
    savefig(p, out)
    println("  ✓ Paper-style figure saved to: $out")
end

# autorun only when invoked directly (julia fit_report.jl <problem.toml>)
if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: julia --project=. fit_report.jl problems/<name>.toml")
    fit_report(ARGS[1])
end
