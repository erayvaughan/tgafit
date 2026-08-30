# =============================================================================
# Generic visualization: forward runs, all-experiments overview, validation fit.
# =============================================================================

using Plots, Printf

const _COLORS = [:blue, :red, :green, :orange, :purple, :brown, :magenta, :gray]
_color(i) = _COLORS[mod1(i, length(_COLORS))]
_explabel(e) = "$(e.id): $(round(e.beta*60, sigdigits=4)) K/min, $(round(100*e.PO2, sigdigits=3))% O₂"

"""
    forward_run(spec; beta_kmin, PO2, label, n_points)

One forward simulation with the nominal parameters; plots total solid mass plus
each solid species vs temperature, and prints a mass-loss summary.
"""
function forward_run(spec::ProblemSpec; beta_kmin::Real, PO2::Real,
                     label::AbstractString = "forward", n_points = 400)
    h = build_system(spec)
    s = simulate(h, spec, beta_kmin / 60.0, PO2; n_points = n_points)

    p = plot(xlabel = "Temperature [K]", ylabel = "Mass Fraction",
             title = @sprintf("%s — %.4g K/min, %.3g%% O₂", spec.name, beta_kmin, 100*PO2),
             legend = :bottomleft, size = (800, 500), dpi = 150, grid = true, framestyle = :box)
    plot!(p, s.temperatures, s.mass, label = "Total solid mass", color = :black, linewidth = 3)
    for (i, sp) in enumerate(solid_species(spec))
        plot!(p, s.temperatures, s.species[sp.name], label = String(sp.name),
              color = _color(i), linewidth = 1.8, linestyle = :dash)
    end
    mkpath("results")
    outfile = "results/$(label).png"
    savefig(p, outfile)

    m0, mend = s.mass[1], s.mass[end]
    println("  ✓ Plot saved to: $outfile")
    println(@sprintf("  Initial solid mass: %.4f   Final residual: %.4f   Mass loss: %.1f%%",
                     m0, mend, 100*(m0 - mend)/m0))
    for thr in (0.95, 0.50)
        idx = findfirst(m -> m <= thr * m0, s.mass)
        idx !== nothing && println(@sprintf("  T at %2.0f%% mass loss: %.1f K", 100*(1-thr), s.temperatures[idx]))
    end
    return p
end

function plot_all_experiments(clean, spec::ProblemSpec)
    p = plot(xlabel = "Temperature [K]", ylabel = "Mass Fraction",
             title = "$(spec.name) — TGA data, all experiments",
             legend = :bottomleft, size = (800, 500), dpi = 150, grid = true, framestyle = :box)
    for (i, e) in enumerate(experiments(spec))
        d = clean[e.id]; c = _color(i)
        scatter!(p, d.temperatures, d.mass_noisy, label = _explabel(e) * " (data)",
                 color = c, alpha = 0.25, markersize = 1.5, markerstrokewidth = 0)
        plot!(p, d.temperatures, d.mass_clean, label = _explabel(e) * " (model)",
              color = c, linewidth = 2)
    end
    savefig(p, "results/all_experiments.png")
    println("  All-experiments plot saved to: results/all_experiments.png")
    return p
end

function plot_validation(h, result, petab_prob, clean, spec::ProblemSpec)
    recovered = Dict(zip(string.(petab_prob.xnames), result.xmin))
    override = Dict{Symbol,Float64}()
    for nm in estimated_names(spec)
        haskey(recovered, string(nm)) && (override[nm] = recovered[string(nm)])
    end

    p = plot(xlabel = "Temperature [K]", ylabel = "Mass Fraction",
             title = "$(spec.name): data vs optimized fit",
             legend = :bottomleft, size = (800, 500), dpi = 150, grid = true, framestyle = :box)
    for (i, e) in enumerate(experiments(spec))
        d = clean[e.id]; c = _color(i)
        scatter!(p, d.temperatures, d.mass_noisy, label = _explabel(e) * " data",
                 color = c, alpha = 0.3, markersize = 2, markerstrokewidth = 0)
        s = simulate(h, spec, e.beta, e.PO2; override = override, n_points = 300)
        plot!(p, s.temperatures, s.mass, label = _explabel(e) * " fit", color = c, linewidth = 2.5)
    end
    savefig(p, "results/validation_plot.png")
    println("\n  Validation plot saved to: results/validation_plot.png")
    return p
end
