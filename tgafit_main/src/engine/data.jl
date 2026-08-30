# =============================================================================
# Data: simulate experiments, generate synthetic TGA data, or load real data
# from CSV. Output is a PEtab-compatible measurement DataFrame in every case.
# =============================================================================

using OrdinaryDiffEq, DataFrames, CSV, Random

"""
    simulate(h, spec, beta_Ks, PO2; override, n_points) -> NamedTuple

Solve one experiment. Returns (times, temperatures, species::Dict, mass) where
`mass` is the total solid mass fraction (sum of non-gas species).
"""
function simulate(h, spec::ProblemSpec, beta_Ks::Real, PO2::Real;
                  override = Dict{Symbol,Float64}(), n_points = spec.n_points)
    u0   = u0_map(h, spec)
    vals = param_values(spec; override = override)
    pmap = Pair{Any,Float64}[h.psym[k] => v for (k, v) in vals]
    push!(pmap, h.psym[:beta] => float(beta_Ks))
    push!(pmap, h.psym[:PO2]  => float(PO2))

    t_final = (spec.T_final - spec.T0) / beta_Ks
    oprob = ODEProblem(h.sys, u0, (0.0, t_final), pmap)
    sol = solve(oprob, Rodas5P(); abstol = 1e-10, reltol = 1e-8, saveat = t_final / n_points)

    species = Dict{Symbol,Vector{Float64}}(s.name => sol[h.spsym[s.name]] for s in spec.species)
    mass = reduce(.+, (species[s.name] for s in solid_species(spec)))
    return (times = sol.t, temperatures = sol[h.spsym[:Temp]], species = species, mass = mass)
end

# --- synthetic data ----------------------------------------------------------

function generate_synthetic(h, spec::ProblemSpec)
    Random.seed!(spec.seed)
    clean = Dict{Symbol,NamedTuple}()
    obs, sim, ts, meas = String[], String[], Float64[], Float64[]

    for e in experiments(spec)
        println("  Simulating $(e.id): β = $(round(e.beta*60, sigdigits=4)) K/min, PO₂ = $(e.PO2)")
        s = simulate(h, spec, e.beta, e.PO2)
        noise = spec.noise_sigma .* randn(length(s.mass))
        mass_noisy = clamp.(s.mass .+ noise, 0.0, 1.0)
        clean[e.id] = (times = s.times, temperatures = s.temperatures,
                       mass_clean = s.mass, mass_noisy = mass_noisy)
        for i in eachindex(s.times)
            push!(obs, "mass_total"); push!(sim, String(e.id))
            push!(ts, s.times[i]);    push!(meas, mass_noisy[i])
        end
    end
    df = DataFrame(obs_id = obs, simulation_id = sim, time = ts, measurement = meas)
    return df, clean
end

# --- real data from CSV ------------------------------------------------------

"""
Load measurements from per-experiment CSV files. Each file must have a
temperature column and a (normalized, 0–1) mass column. Time is reconstructed
from temperature via the linear heating program: t = (T − T0) / β.
"""
function load_csv_data(spec::ProblemSpec)
    clean = Dict{Symbol,NamedTuple}()
    obs, sim, ts, meas = String[], String[], Float64[], Float64[]

    for e in experiments(spec)
        isfile(e.file) || error("CSV file not found for $(e.id): $(e.file)")
        df = CSV.read(e.file, DataFrame)
        T = Float64.(df[!, spec.temp_col])
        m = Float64.(df[!, spec.mass_col])
        # clamp at 0: data points a hair below T0 (binning jitter) would otherwise
        # produce negative times, which PEtab rejects
        time = max.((T .- spec.T0) ./ e.beta, 0.0)
        order = sortperm(time)
        time, T, m = time[order], T[order], m[order]
        println("  Loaded $(e.id): $(nrow(df)) points from $(e.file)")
        clean[e.id] = (times = time, temperatures = T, mass_clean = m, mass_noisy = m)
        for i in eachindex(time)
            push!(obs, "mass_total"); push!(sim, String(e.id))
            push!(ts, time[i]);       push!(meas, m[i])
        end
    end
    df = DataFrame(obs_id = obs, simulation_id = sim, time = ts, measurement = meas)
    return df, clean
end

"""Dispatch on the configured data mode."""
function get_measurements(h, spec::ProblemSpec)
    spec.data_mode == :synthetic && return generate_synthetic(h, spec)
    spec.data_mode == :csv       && return load_csv_data(spec)
    error("Unknown data.mode = $(spec.data_mode) (expected :synthetic or :csv).")
end
