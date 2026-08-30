# =============================================================================
# Generic entry point for the decomposition-kinetics solver.
#
#   julia --project=. solve.jl problems/onera.toml
#   julia --project=. solve.jl problems/onera.toml --forward 10 0.21
#
# Or interactively:
#   include("src/engine/engine.jl")
#   spec = load_problem("problems/onera.toml")
#   solve_problem(spec)
# =============================================================================

include("src/engine/engine.jl")

function main(args)
    isempty(args) && error("usage: julia --project=. solve.jl problems/<name>.toml [--forward beta PO2]")
    cfgpath = args[1]
    spec = load_problem(cfgpath)

    if length(args) >= 2 && args[2] == "--forward"
        beta = length(args) >= 3 ? parse(Float64, args[3]) : spec.exp_betas[1]
        po2  = length(args) >= 4 ? parse(Float64, args[4]) : spec.exp_PO2s[1]
        forward_run(spec; beta_kmin = beta, PO2 = po2, label = "forward")
    else
        solve_problem(spec)
    end
end

main(ARGS)
