# =============================================================================
# Generic forward-model construction: turn a ProblemSpec into a Catalyst
# ReactionSystem with dynamically-named species and parameters.
# =============================================================================

using Catalyst, ModelingToolkit

const _POW_FLOOR = 1e-10   # floor on any base raised to an estimated power (ForwardDiff-safe)
const _EPS       = 1e-30   # guards division by a zero available amount

"""
    build_system(spec) -> NamedTuple(sys, spsym, psym, avail)

Construct the reaction system. `spsym`/`psym` map config names (Symbols) to the
symbolic species/parameters, so the rest of the engine can build u0/parameter
maps generically. Parameter names: log10_A_<rxn>, E_<rxn>, n_<rxn>, m_<rxn>
(aerobic only), yields nu_<rxn>_<product>, plus the conditions PO2 and beta.
"""
function build_system(spec::ProblemSpec)
    t = default_t()
    mk_sp(nm) = only(@species $(nm)(t))
    mk_p(nm)  = only(@parameters $(nm))

    # species (+ temperature as a dynamic variable)
    spsym = Dict{Symbol,Any}(:Temp => mk_sp(:Temp))
    for s in spec.species
        spsym[s.name] = mk_sp(s.name)
    end

    psym = Dict{Symbol,Any}()
    getp!(nm) = get!(() -> mk_p(nm), psym, nm)
    beta = getp!(:beta)
    PO2  = getp!(:PO2)

    avail = availability(spec)
    R, PO2ref = spec.R_gas, spec.PO2_ref
    Temp = spsym[:Temp]

    # Temperature ramp: dT/dt = β (linear heating program)
    rxns = Any[Reaction(beta, nothing, [Temp], nothing, [1])]

    for r in spec.reactions
        rn  = r.name
        lA  = getp!(Symbol("log10_A_$rn"))
        E   = getp!(Symbol("E_$rn"))
        n   = getp!(Symbol("n_$rn"))
        s   = spsym[r.reactant]
        a   = avail[r.reactant]                       # constant available amount
        frac = max(s, 0.0) / (a + _EPS)               # (1 − α) = reactant / available
        rate = a * (10.0^lA) * exp(-E / (R * Temp)) * max(frac, _POW_FLOOR)^n
        if r.aerobic
            m = getp!(Symbol("m_$rn"))
            rate = rate * max(PO2 / PO2ref, _POW_FLOOR)^m
        end
        prods = [spsym[p] for p in r.products]
        yps   = [getp!(Symbol("nu_$(rn)_$(p)")) for p in r.products]  # yields are params
        push!(rxns, Reaction(rate, [s], prods, [1], yps; only_use_rate=true))
    end

    # Pass species/parameters explicitly: species that appear in no reaction
    # (inert residues, common in real TGA data) would otherwise be dropped.
    sts = [spsym[:Temp]; [spsym[s.name] for s in spec.species]]
    ps  = collect(values(psym))
    @named sys = ReactionSystem(rxns, t, sts, ps)
    sys = complete(sys)
    return (sys = sys, spsym = spsym, psym = psym, avail = avail)
end

"""
    param_values(spec; override) -> Dict{Symbol,Float64}

Nominal numeric value for every model parameter (kinetics + yields), keyed by
parameter name. `override` replaces estimated kinetics with recovered values.
PO2/beta are set per-experiment, not here.
"""
function param_values(spec::ProblemSpec; override = Dict{Symbol,Float64}())
    d = Dict{Symbol,Float64}()
    for r in spec.reactions
        d[Symbol("log10_A_$(r.name)")] = r.log10_A
        d[Symbol("E_$(r.name)")]       = r.E
        d[Symbol("n_$(r.name)")]       = r.n
        r.aerobic && (d[Symbol("m_$(r.name)")] = r.m)
        for (p, y) in zip(r.products, r.yields)
            d[Symbol("nu_$(r.name)_$(p)")] = y
        end
    end
    for (k, v) in override
        d[k] = v
    end
    return d
end

"""u0 pairs (species => initial), with temperature at T0."""
function u0_map(h, spec::ProblemSpec)
    u = Pair{Any,Float64}[h.spsym[:Temp] => spec.T0]
    for s in spec.species
        push!(u, h.spsym[s.name] => s.initial)
    end
    return u
end
