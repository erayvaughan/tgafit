# =============================================================================
# Problem specification: a config-driven description of a multi-stage
# thermal-decomposition kinetics problem (any species / any reactions).
#
# A problem is fully defined by a TOML file (see problems/*.toml). Nothing about
# a specific material is hardcoded here — the ONERA CFRP case is just one config.
#
# Rate-law family (standard TGA, conversion-based, paper Eq. 11):
#   dα_m/dt = f_m(O₂) · (1 − α_m)^n_m · A_m · exp(−E_m / (R·T))
# with (1 − α_m) = reactant / available_amount and the oxygen factor
#   f_m(O₂) = (PO₂ / PO₂_ref)^m_m   (aerobic),   1   (anaerobic).
# =============================================================================

using TOML

struct SpeciesSpec
    name::Symbol
    initial::Float64   # initial mass fraction
    is_gas::Bool       # gases are excluded from the observable solid mass
end

struct ReactionSpec
    name::Symbol
    reactant::Symbol            # the species whose conversion drives the reaction
    products::Vector{Symbol}
    yields::Vector{Float64}     # stoichiometric mass coefficients (fixed, ≈ sum 1)
    aerobic::Bool               # requires O₂ (oxidation) or not (pyrolysis)
    log10_A::Float64            # nominal Arrhenius pre-exponential (log10, 1/s)
    E::Float64                  # nominal activation energy [J/mol]
    n::Float64                  # nominal reaction order
    m::Float64                  # nominal oxygen order (used only if aerobic)
    estimate::Vector{Symbol}    # subset of (:log10_A,:E,:n,:m) to estimate
end

struct ProblemSpec
    name::String
    T0::Float64
    T_final::Float64
    R_gas::Float64
    PO2_ref::Float64
    species::Vector{SpeciesSpec}
    reactions::Vector{ReactionSpec}
    exp_ids::Vector{Symbol}
    exp_betas::Vector{Float64}   # K/min (user-facing)
    exp_PO2s::Vector{Float64}
    exp_files::Vector{String}    # CSV paths (csv mode); empty otherwise
    data_mode::Symbol            # :synthetic or :csv
    noise_sigma::Float64
    n_points::Int
    seed::Int
    n_multistarts::Int
    temp_col::String
    mass_col::String
end

# --- small helpers -----------------------------------------------------------

solid_species(spec::ProblemSpec) = [s for s in spec.species if !s.is_gas]
species_names(spec::ProblemSpec)  = [s.name for s in spec.species]
isaerobic_problem(spec) = any(r.aerobic for r in spec.reactions)

"""Experiments as named tuples; β converted from K/min to K/s (kinetics use s)."""
function experiments(spec::ProblemSpec)
    return [(id = spec.exp_ids[i], beta = spec.exp_betas[i] / 60.0,
             PO2 = spec.exp_PO2s[i],
             file = i <= length(spec.exp_files) ? spec.exp_files[i] : "")
            for i in eachindex(spec.exp_ids)]
end

"""
    availability(spec) -> Dict{Symbol,Float64}

Total amount of each reactant that ever exists = initial + everything produced
upstream, resolved through the production graph (paper's (ρ_m)_Σ). This is the
denominator of (1 − α) = reactant / available. Valid as a constant because TGA
stages are temperature-separated (a reactant is fully formed before it reacts).
"""
function availability(spec::ProblemSpec)
    init  = Dict(s.name => s.initial for s in spec.species)
    avail = Dict{Symbol,Float64}()
    visiting = Set{Symbol}()
    function resolve(s::Symbol)
        haskey(avail, s) && return avail[s]
        s in visiting && error("Cyclic production detected involving species $s")
        push!(visiting, s)
        total = get(init, s, 0.0)
        for r in spec.reactions, (p, y) in zip(r.products, r.yields)
            p == s && (total += y * resolve(r.reactant))
        end
        delete!(visiting, s)
        avail[s] = total
        return total
    end
    for r in spec.reactions
        resolve(r.reactant)
    end
    return avail
end

# --- TOML loading ------------------------------------------------------------

_sym(x) = Symbol(String(x))

function load_problem(path::AbstractString)
    d = TOML.parsefile(path)

    grid = get(d, "grid", Dict{String,Any}())
    species = SpeciesSpec[]
    for s in get(d, "species", [])
        push!(species, SpeciesSpec(_sym(s["name"]), Float64(s["initial"]),
                                   Bool(get(s, "gas", false))))
    end

    reactions = ReactionSpec[]
    for r in get(d, "reaction", [])
        prods  = Symbol[_sym(p) for p in r["products"]]
        yields = Float64[Float64(y) for y in r["yields"]]
        est    = Symbol[_sym(e) for e in get(r, "estimate", String[])]
        push!(reactions, ReactionSpec(
            _sym(r["name"]), _sym(r["reactant"]), prods, yields,
            Bool(get(r, "aerobic", false)),
            Float64(r["log10_A"]), Float64(r["E"]), Float64(r["n"]),
            Float64(get(r, "m", 0.0)), est))
    end

    ex = get(d, "experiments", Dict{String,Any}())
    ids   = Symbol[_sym(x) for x in get(ex, "ids", String[])]
    betas = Float64[Float64(x) for x in get(ex, "beta", Float64[])]
    po2s  = Float64[Float64(x) for x in get(ex, "PO2", Float64[])]
    files = String[String(x) for x in get(ex, "files", String[])]

    dat = get(d, "data", Dict{String,Any}())
    est = get(d, "estimation", Dict{String,Any}())

    return ProblemSpec(
        String(get(d, "name", "unnamed problem")),
        Float64(get(grid, "T0", 300.0)),
        Float64(get(grid, "T_final", 1200.0)),
        Float64(get(grid, "R_gas", 8.314)),
        Float64(get(grid, "PO2_ref", 0.21)),
        species, reactions, ids, betas, po2s, files,
        _sym(get(dat, "mode", "synthetic")),
        Float64(get(dat, "noise_sigma", 0.005)),
        Int(get(dat, "n_points", 150)),
        Int(get(dat, "seed", 42)),
        Int(get(est, "n_multistarts", 10)),
        String(get(dat, "temp_col", "temperature")),
        String(get(dat, "mass_col", "mass")),
    )
end

# --- validation --------------------------------------------------------------

"""
    validate_problem(spec) -> (errors, warnings)

Errors block the run; warnings are advisory (mass balance, identifiability).
"""
function validate_problem(spec::ProblemSpec)
    errors   = String[]
    warnings = String[]
    names = Set(species_names(spec))

    isempty(spec.species)   && push!(errors, "At least one species is required.")
    isempty(spec.reactions) && push!(errors, "At least one reaction is required.")

    nexp = length(spec.exp_ids)
    nexp == 0 && push!(errors, "At least one experiment is required.")
    (length(spec.exp_betas) == nexp && length(spec.exp_PO2s) == nexp) ||
        push!(errors, "Experiment vectors (ids/beta/PO2) have inconsistent lengths.")
    length(unique(spec.exp_ids)) == nexp || push!(errors, "Experiment IDs must be unique.")
    if spec.data_mode == :csv && length(spec.exp_files) != nexp
        push!(errors, "data.mode = \"csv\" requires one file per experiment.")
    end

    for r in spec.reactions
        r.reactant in names || push!(errors, "Reaction $(r.name): unknown reactant $(r.reactant).")
        length(r.products) == length(r.yields) ||
            push!(errors, "Reaction $(r.name): products and yields differ in length.")
        for p in r.products
            p in names || push!(errors, "Reaction $(r.name): unknown product $p.")
        end
        sy = sum(r.yields)
        (0.9 <= sy <= 1.1) ||
            push!(warnings, "Reaction $(r.name): yields sum to $(round(sy,digits=3)) (expected ≈ 1, mass balance).")
        if r.aerobic && !(:m in r.estimate)
            # fine, just fixed
        end
        for e in r.estimate
            e in (:log10_A, :E, :n, :m) ||
                push!(errors, "Reaction $(r.name): cannot estimate unknown parameter $e.")
            e == :m && !r.aerobic &&
                push!(warnings, "Reaction $(r.name): estimating m but reaction is anaerobic (no O₂ term).")
        end
    end

    # a species consumed by >1 reaction makes the per-reaction (1−α) ambiguous
    consumed = Dict{Symbol,Int}()
    for r in spec.reactions
        consumed[r.reactant] = get(consumed, r.reactant, 0) + 1
    end
    for (s, c) in consumed
        c > 1 && push!(warnings, "Species $s is consumed by $c reactions; the conversion " *
                                 "approximation assumes a single consumer per species.")
    end

    total0 = sum(s.initial for s in spec.species; init = 0.0)
    (0.95 <= total0 <= 1.05) ||
        push!(warnings, "Initial mass fractions sum to $(round(total0,digits=3)) (expected ≈ 1).")

    for i in 1:min(nexp, length(spec.exp_betas), length(spec.exp_PO2s))
        spec.exp_betas[i] > 0 || push!(errors, "$(spec.exp_ids[i]): heating rate β must be > 0.")
        (0.0 < spec.exp_PO2s[i] <= 1.0) || push!(errors, "$(spec.exp_ids[i]): PO₂ must be in (0, 1].")
    end
    spec.T_final > spec.T0 || push!(errors, "T_final must be greater than T0.")

    if nexp >= 1 && length(unique(spec.exp_PO2s)) == 1 && isaerobic_problem(spec)
        push!(warnings, "All experiments share the same PO₂ — oxygen orders (m) are " *
                        "unidentifiable. Add an experiment at a different PO₂.")
    end
    if nexp >= 1 && length(unique(spec.exp_betas)) == 1
        push!(warnings, "All experiments share the same heating rate — activation energies " *
                        "are poorly constrained (A–E correlation). Vary β across experiments.")
    end

    return errors, warnings
end
