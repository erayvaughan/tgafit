# =============================================================================
# Generic inverse problem: build the PEtab model from a ProblemSpec, estimate
# the requested parameters, and compare recovered vs nominal values.
# =============================================================================

using PEtab, OrdinaryDiffEq, OptimizationOptimJL, DataFrames, Printf, Random, Statistics
using OptimizationOptimJL.Optim: LBFGS

# Estimation bounds derived around each nominal value.
_b_log10A(v) = (v - 3.0, v + 3.0)
_b_E(v)      = (0.6 * v, 1.5 * v)
_b_n(v)      = (max(0.1, v - 1.0), v + 1.5)
_b_m(v)      = (0.05, v + 1.2)

_bounds_for(kind, v) = kind === :log10_A ? _b_log10A(v) :
                       kind === :E       ? _b_E(v)      :
                       kind === :n       ? _b_n(v)      : _b_m(v)

"""
    estimated_names(spec) -> Vector{Symbol}

Model-parameter names that are being estimated (reaction param × estimate flag).
"""
function estimated_names(spec::ProblemSpec)
    names = Symbol[]
    for r in spec.reactions, k in r.estimate
        push!(names, Symbol("$(k)_$(r.name)"))
    end
    return names
end

"""
    setup_petab(h, spec, measurements_df) -> PEtabODEProblem
"""
function setup_petab(h, spec::ProblemSpec, measurements_df)
    # --- Observable: total solid mass = Σ solid species, constant Gaussian noise ---
    @parameters sigma_mass
    solid_syms = [h.spsym[s.name] for s in solid_species(spec)]
    mass_expr  = reduce(+, solid_syms)
    observables = [PEtabObservable(:mass_total, mass_expr, sigma_mass)]

    # --- Simulation conditions: per-experiment β and PO₂ ---
    beta, PO2 = h.psym[:beta], h.psym[:PO2]
    simulation_conditions = [PEtabCondition(e.id, beta => e.beta, PO2 => e.PO2)
                             for e in experiments(spec)]

    # --- Parameters to estimate (per reaction × estimate flags) + noise ---
    petab_parameters = PEtabParameter[]
    for r in spec.reactions
        nominal = Dict(:log10_A => r.log10_A, :E => r.E, :n => r.n, :m => r.m)
        for k in r.estimate
            lb, ub = _bounds_for(k, nominal[k])
            push!(petab_parameters,
                  PEtabParameter(Symbol("$(k)_$(r.name)"), value = nominal[k],
                                 lb = lb, ub = ub, scale = :lin))
        end
    end
    sigma_start = max(spec.noise_sigma, 1.1e-4)
    push!(petab_parameters,
          PEtabParameter(:sigma_mass, value = sigma_start, lb = 1e-4, ub = 0.1, scale = :log10))

    # --- Initial conditions (same for all experiments) ---
    state_map = u0_map(h, spec)

    # --- Fixed parameters: yields + any kinetic param NOT estimated ---
    est = Set(estimated_names(spec))
    parameter_map = Pair{Any,Float64}[]
    for (k, v) in param_values(spec)
        sym = h.psym[k]
        (k in est) && continue          # estimated, not fixed
        push!(parameter_map, sym => v)
    end

    model = PEtabModel(h.sys, observables, measurements_df, petab_parameters;
                       simulation_conditions = simulation_conditions,
                       speciemap = state_map, parametermap = parameter_map,
                       verbose = true)

    return PEtabODEProblem(model;
        odesolver = ODESolver(Rodas5P(); abstol = 1e-8, reltol = 1e-6,
                              maxiters = Int64(1e5), force_dtmin = true),
        gradient_method = :ForwardDiff, hessian_method = :ForwardDiff)
end

"""
    run_estimation(petab_prob; n_multistarts) -> (xmin, fmin, runs)

Multistart LBFGS: nominal start + controlled ±20% perturbations within bounds
(avoids the Arrhenius overflow that random Latin-hypercube sampling causes).
"""
function run_estimation(petab_prob; n_multistarts = 10)
    println("\n" * "="^60)
    println("  Multistart optimization ($n_multistarts starts)")
    println("="^60)
    Random.seed!(123)
    nominal = collect(petab_prob.xnominal_transformed)
    lb = collect(petab_prob.lower_bounds)
    ub = collect(petab_prob.upper_bounds)

    runs = []; best_f = Inf; best_x = nominal
    for i in 1:n_multistarts
        x0 = i == 1 ? copy(nominal) :
             clamp.(nominal .+ 0.2 .* (ub .- lb) .* (rand(length(nominal)) .- 0.5), lb, ub)
        println("  Start $i/$n_multistarts...")
        try
            res = calibrate(petab_prob, x0, LBFGS())
            println("    → NLL = $(round(res.fmin, digits=2))")
            res.fmin < best_f && (best_f = res.fmin; best_x = res.xmin)
            push!(runs, res)
        catch e
            msg = sprint(showerror, e)
            println("    → Failed: ", first(msg, 160))
        end
    end
    isempty(runs) && @warn "All starts failed — returning nominal parameters."
    println("\n  Best NLL: $(round(best_f, digits=2))")
    return (xmin = best_x, fmin = best_f, runs = runs)
end

"""
    compare(result, petab_prob, spec) -> DataFrame

True (nominal) vs recovered for every estimated parameter, with % error.
"""
function compare(result, petab_prob, spec::ProblemSpec)
    recovered = Dict(zip(string.(petab_prob.xnames), result.xmin))
    nominal = param_values(spec)
    names, tv, rv, pe = String[], Float64[], Float64[], Float64[]
    for nm in estimated_names(spec)
        s = string(nm)
        haskey(recovered, s) || continue
        t = nominal[nm]; r = recovered[s]
        push!(names, s); push!(tv, t); push!(rv, r)
        push!(pe, abs(r - t) / max(abs(t), eps()) * 100.0)
    end
    return DataFrame(Parameter = names, True_Value = tv,
                     Recovered = round.(rv, sigdigits = 5), Pct_Error = round.(pe, digits = 2))
end

function print_comparison(df, data_mode::Symbol = :synthetic)
    twin = data_mode == :synthetic   # twin experiment: nominal really is "true"
    println("\n" * "="^72)
    if twin
        println("  Parameter Estimation: True (nominal) vs Recovered")
    else
        println("  Parameter Estimation: Initial guess vs Fitted (real data)")
        println("  (\"% Moved\" is how far the optimizer moved from the DTG-based")
        println("   starting guess — NOT an error. Fit quality = RMSE/R² vs data.)")
    end
    println("="^72)
    @printf("  %-22s  %12s  %12s  %10s\n", "Parameter",
            twin ? "True" : "Init guess", twin ? "Recovered" : "Fitted",
            twin ? "% Error" : "% Moved")
    println("  " * "-"^60)
    for row in eachrow(df)
        @printf("  %-22s  %12.4g  %12.4g  %9.2f%%\n",
                row.Parameter, row.True_Value, row.Recovered, row.Pct_Error)
    end
    println("="^72)
    label = twin ? "Mean % Error" : "Mean % Moved"
    nrow(df) > 0 && println("  $label: $(round(mean(df.Pct_Error), digits=2))%")
    println("="^72)
end
