# =============================================================================
# Engine: end-to-end pipeline for any multi-stage decomposition kinetics problem
# defined by a ProblemSpec (loaded from a TOML config).
#
#   include("src/engine/engine.jl")
#   spec = load_problem("problems/onera.toml")
#   solve_problem(spec)                       # full inverse workflow
#   forward_run(spec; beta_kmin=10, PO2=0.21) # single forward simulation
# =============================================================================

const _ENGINE_DIR = @__DIR__
include(joinpath(_ENGINE_DIR, "spec.jl"))
include(joinpath(_ENGINE_DIR, "build.jl"))
include(joinpath(_ENGINE_DIR, "data.jl"))
include(joinpath(_ENGINE_DIR, "inverse.jl"))
include(joinpath(_ENGINE_DIR, "viz.jl"))

using CSV, DataFrames

"""
    solve_problem(spec; skip_estimation=false) -> NamedTuple

Run the full pipeline:
  1. build the forward model
  2. get measurements (synthetic or CSV)
  3. set up the PEtab inverse problem
  4. estimate parameters, compare, and plot the validation fit
"""
function solve_problem(spec::ProblemSpec; skip_estimation::Bool = false)
    println("="^64)
    println("  Decomposition kinetics solver — $(spec.name)")
    println("="^64)

    errors, warnings = validate_problem(spec)
    for w in warnings; println("  ⚠ $w"); end
    if !isempty(errors)
        for e in errors; println("  ✗ $e"); end
        error("Problem specification has $(length(errors)) error(s); aborting.")
    end
    mkpath("results")

    println("\n📌 Building forward model...")
    h = build_system(spec)
    println("  ✓ $(length(reactions(h.sys))) reactions, $(length(species(h.sys))) species, " *
            "$(length(parameters(h.sys))) parameters")

    println("\n📌 Acquiring measurements ($(spec.data_mode))...")
    measurements_df, clean = get_measurements(h, spec)
    println("  ✓ $(nrow(measurements_df)) points across $(length(spec.exp_ids)) experiment(s)")
    CSV.write("results/measurements.csv", measurements_df)
    plot_all_experiments(clean, spec)

    if skip_estimation
        println("\n  (skip_estimation = true — stopping after data stage)")
        return (h = h, measurements = measurements_df, clean = clean)
    end

    println("\n📌 Setting up PEtab inverse problem...")
    petab_prob = setup_petab(h, spec, measurements_df)
    println("  ✓ Estimating: ", join(string.(petab_prob.xnames), ", "))

    println("\n📌 Estimating parameters...")
    result = run_estimation(petab_prob; n_multistarts = spec.n_multistarts)
    cmp = compare(result, petab_prob, spec)
    print_comparison(cmp, spec.data_mode)
    CSV.write("results/parameter_comparison.csv", cmp)
    plot_validation(h, result, petab_prob, clean, spec)

    println("\n" * "="^64)
    println("  ✓ Done — results/ has measurements, comparison, and plots")
    println("="^64)
    return (h = h, measurements = measurements_df, clean = clean,
            petab = petab_prob, result = result, comparison = cmp)
end
