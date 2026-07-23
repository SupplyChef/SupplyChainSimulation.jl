
#import Distributions
using Dates

"""
    metrics_cost_function(s::State)

An alternative to hand-rolling `s -> -get_total_sales(s) + get_total_lost_sales(s) + ...`
as a `cost_function` for `optimize!`: reads `s.metrics` (see `SimMetrics`) instead of
scanning `s`'s `historical_*` arrays the way the equivalent `get_total_*` functions in
Reporting.jl do. Since `SimMetrics` is always kept up to date regardless of
`Env.record_history`, this is the cost function to pass (together with
`record_history=false`) to get `optimize!`'s full per-trial-simulation speedup.

Not `optimize!`'s default: its default is left exactly as it always was (scanning
history, with `record_history` defaulting to `true`) so existing callers - including
ones whose `cost_function` reads `get_total_*` directly - see byte-for-byte identical
behavior. Summing the same costs in a different order (event order here, Set/Dict
iteration order for the `get_total_*` scan) is only equal up to floating-point
rounding, not bit-for-bit, and `optimize!`'s search is sensitive enough to that
- many float comparisons deciding which candidate "wins" over thousands of
evaluations - that swapping the default out from under existing callers could shift
what they converge to. Opt in explicitly once you don't need exact reproducibility
against a pre-existing baseline.
"""
metrics_cost_function(s) = -s.metrics.sales + s.metrics.lost_sales + s.metrics.holding_costs + s.metrics.trip_fixed_costs + s.metrics.trip_unit_costs + 0.001 * s.metrics.orders

function minimize!(lane_policies, policies, envs::Array{Env, 1}, initial_states::Array{State, 1}, x::AbstractVector{Float64}; cost_function)
    i = 1
    for policy in policies
        set_parameters!(policy, x[i:i+length(get_parameters(policy))-1])
        i = i + length(get_parameters(policy))
    end

    value = 0
    for i in 1:length(initial_states)
        # Reset the state's mutable containers in place rather than paying for a
        # fresh deepcopy of the (read-only) supply chain on every evaluation -
        # simulate() then simulates initial_states[i] directly and returns it.
        reset!(initial_states[i])
        final_state = simulate(envs[i], lane_policies, initial_states[i])

        #println(final_state)

        #println("$x $(get_total_lost_sales(final_state)) $(get_total_orders(final_state))")
        value += cost_function(final_state)
    end
    return value
end

"""
    optimize!(supplychain::SupplyChain, lane_policies, initial_states...; cost_function, record_history, method, cma_es_options)

    Optimizes the inventory policies in the supply chain by simulating the inventory movement starting from the initial states and costing the results with the cost function.

    `record_history` controls whether each of the (typically thousands of) trial
    simulations `optimize!` runs archives its full per-period history (see
    `Env.record_history`). It defaults to `true`, matching every trial simulation's
    behavior before `record_history` existed, so existing callers - including custom
    `cost_function`s that call the history-scanning `get_total_*` functions in
    Reporting.jl - keep working unchanged. Pass `record_history=false` once your
    `cost_function` only reads `state.metrics` (see `SimMetrics` and
    `metrics_cost_function`) to skip that per-trial bookkeeping.

    This is independent of `BackwardCoverageOrderingPolicy`: it reads
    `state.outbound_order_quantities` instead of the history-scanning
    `get_total_*` functions, and that index is maintained regardless of
    `record_history` (see `required_lookback`/`Env.needs_outbound_order_index`),
    so `record_history=false` is safe with that policy too.

    `method` selects the search algorithm used to minimize `cost_function` over the
    policies' parameters:
    - `:custom` (the default): the original hand-rolled differential-evolution-style
      optimizer in this file (`bboptimize`/`params`). Kept as the default so existing
      callers see byte-for-byte identical behavior.
    - `:cma_es`: Covariance Matrix Adaptation Evolution Strategy, via the
      CMAEvolutionStrategy.jl package. Tuned via `cma_es_options` (see below) instead
      of `params`, since CMA-ES's options aren't all `Float64` (bounds are vectors,
      some are integers) and `params` is typed `Dict{Symbol, Float64}`.
    - `:nelder_mead`: multi-start Nelder-Mead simplex search, via Optim.jl. Tuned via
      `nelder_mead_options` (see below). A deliberately different kind of algorithm
      from the other two (a deterministic local search with restarts, vs. their
      population-based stochastic search) - worth trying because
      benchmark/compare_optimizers.jl showed :cma_es converging (and then plateauing)
      well before exhausting its evaluation budget on the newsvendor benchmark,
      suggesting a cheaper local refinement with restarts to escape local optima
      might match its quality in even less time, or find a better optimum with the
      leftover budget spent on more restarts instead of one long population search.

    `cma_es_options` (only used when `method === :cma_es`) accepts:
    - `:lower`, `:upper`: box constraints, each either a scalar (broadcast to every
      parameter) or a per-parameter vector. Default `0.0`/`5000.0`, matching
      `:custom`'s default `:SearchRange`.
    - `:maxfevals`: evaluation budget. Default `15000`, matching `:custom`'s default
      `:MaxFuncEvals`, so the two methods are comparable under the same budget.
    - `:sigma0`: initial global step size (CMA-ES's `s0`). Defaults to a quarter of
      the bounds' range, a commonly-used rule of thumb when there's no prior on
      where in the box the optimum sits.
    - `:popsize`, `:seed`, `:verbosity`: passed straight through to
      `CMAEvolutionStrategy.minimize` when given; otherwise left at that package's
      own defaults (`:verbosity` defaults to `0` here instead, since `:custom`
      already prints its own progress and CMA-ES's is redundant in that context).

    `nelder_mead_options` (only used when `method === :nelder_mead`) accepts:
    - `:lower`, `:upper`: box constraints, same shape/defaults as `cma_es_options`.
      Plain Nelder-Mead is unconstrained, so out-of-box candidates are clamped back
      into range before being costed, rather than using Optim.jl's `Fminbox` (a
      log-barrier wrapper built for gradient-based methods, not a natural fit for a
      derivative-free simplex search).
    - `:maxfevals`: total evaluation budget across all restarts combined. Default
      `15000`, matching the other two methods.
    - `:restarts`: number of *additional* Nelder-Mead runs beyond the one seeded at
      `x0`, each getting an even share of `:maxfevals`. Default `5` (6 runs total).
      Each restart perturbs the current incumbent best by a random step (see
      `:restart_scale`), not a fresh uniformly random point in the whole box - see
      `nelder_mead_optimize`'s docstring for why a global-random-restart version of
      this was actively harmful on a 6-parameter problem.
    - `:restart_scale`: restart perturbation size, as a fraction of `:upper - :lower`
      in each dimension. Default `0.2`.
"""
function optimize!(lane_policies, supplychains...; params::Dict{Symbol, Float64}=Dict{Symbol, Float64}(), cost_function=s->-get_total_sales(s) + get_total_lost_sales(s) + get_total_holding_costs(s) + get_total_trip_fixed_costs(s) + get_total_trip_unit_costs(s) + 0.001 * get_total_orders(s), record_history::Bool=true, method::Symbol=:custom, cma_es_options::Dict{Symbol, Any}=Dict{Symbol, Any}(), nelder_mead_options::Dict{Symbol, Any}=Dict{Symbol, Any}())
    initial_states = State.(supplychains)
    envs = [Env(supplychain, initial_states, lane_policies; record_history=record_history) for supplychain in supplychains]

    # Iterate lane_policies in a deterministic order (by lane/product name)
    # rather than raw Dict key order: keys(lane_policies) iterates in hash
    # bucket order, which for id-hashed Lane/Product keys depends on
    # construction order across the whole process, not just this network's
    # structure. Since this order determines which slice of the optimizer's
    # parameter vector gets assigned to which policy (see minimize! above),
    # letting it vary non-reproducibly changes which local optimum the
    # search converges to.
    sorted_keys = sort(collect(keys(lane_policies)); by = k -> (string(k[1]), k[2].name))
    policies = unique([lane_policies[k] for k in sorted_keys])
    #println(policies)

    x0 = vcat([get_parameters(policy) for policy in policies]...)
    x0 = convert(Array{Float64, 1}, x0)

    f = x -> minimize!(lane_policies, policies, collect(envs), collect(initial_states), x; cost_function=cost_function)

    if method === :custom
        # :MaxStepsWithoutProgress is deliberately left unset here (unless
        # params overrides it) rather than hardcoded - bboptimize derives a
        # default from both :MaxFuncEvals and its own pool_size, since the
        # two interact (see bboptimize's comment on that default).
        max_func_evals = get(params, :MaxFuncEvals, 15000.0)
        best = SupplyChainSimulation.bboptimize(f,
                         x0,
                        merge(Dict(:MaxFuncEvals => max_func_evals,
                             :SearchRange => (-0.0, 5000.0),
                             :NumDimensions => length(x0)), params))
    elseif method === :cma_es
        best = cma_es_optimize(f, x0, cma_es_options)
    elseif method === :nelder_mead
        best = nelder_mead_optimize(f, x0, nelder_mead_options)
    else
        error("optimize!: unknown method $(repr(method)); supported methods are :custom, :cma_es, and :nelder_mead")
    end

    i = 1
    for policy in policies
        set_parameters!(policy, best[i:i+length(get_parameters(policy))-1])
        i = i + length(get_parameters(policy))
    end
end

"""
    cma_es_optimize(f, x0, options)

Minimizes `f` starting from `x0` with CMAEvolutionStrategy.jl, translating this
package's `optimize!(...; method=:cma_es, cma_es_options=...)` options (see
`optimize!`'s docstring) into `CMAEvolutionStrategy.minimize`'s keyword arguments.
Split out of `optimize!` so the CMA-ES-specific option handling (scalar-vs-vector
bounds, the `sigma0` default, forwarding `popsize`/`seed` only when given) doesn't
clutter the method-dispatch branch above.
"""
function cma_es_optimize(f, x0::Array{Float64, 1}, options::Dict{Symbol, Any})
    n = length(x0)
    lower = get(options, :lower, 0.0)
    upper = get(options, :upper, 5000.0)
    lower_bounds = lower isa AbstractVector ? convert(Array{Float64, 1}, lower) : fill(convert(Float64, lower), n)
    upper_bounds = upper isa AbstractVector ? convert(Array{Float64, 1}, upper) : fill(convert(Float64, upper), n)

    sigma0 = get(options, :sigma0, (upper_bounds[1] - lower_bounds[1]) / 4)

    minimize_kwargs = Dict{Symbol, Any}(
        :lower => lower_bounds,
        :upper => upper_bounds,
        :maxfevals => get(options, :maxfevals, 15000),
        :verbosity => get(options, :verbosity, 0),
    )
    haskey(options, :popsize) && (minimize_kwargs[:popsize] = options[:popsize])
    haskey(options, :seed) && (minimize_kwargs[:seed] = options[:seed])

    result = CMAEvolutionStrategy.minimize(f, x0, sigma0; minimize_kwargs...)
    return CMAEvolutionStrategy.xbest(result)
end

"""
    nelder_mead_optimize(f, x0, options)

Minimizes `f` with iterated-local-search-style multi-start Nelder-Mead (Optim.jl),
translating this package's `optimize!(...; method=:nelder_mead, nelder_mead_options=...)`
options (see `optimize!`'s docstring). One run is seeded at `x0`; each of
`options[:restarts]` more perturbs the *current incumbent best* by a random step
scaled to `options[:restart_scale]` times the box's range in each dimension, rather
than drawing a fresh uniformly random point from the whole box. Each run gets an
even share of `:maxfevals`; the best result across all runs wins.

Earlier versions of this function drew every restart uniformly from the whole
`[lower, upper]` box. benchmark/compare_optimizers.jl's beer_game comparison (6
parameters, vs. newsvendor's 2) showed this was actively harmful, not just
inefficient: a uniform sample of a 6-dimensional box is overwhelmingly likely to
land far from any good region, so nearly every restart wasted its whole budget
converging to a poor local optimum instead of ever improving on the x0-seeded run -
visible as wildly inconsistent, often much worse than :custom/:cma_es results.
Perturbing around the incumbent instead keeps restarts a local
refinement/escape-a-nearby-local-optimum mechanism, which does not need to get
harder as dimensionality grows the way "land anywhere useful in the full box"
does - at the cost of being less able to find a *distant* better region than a
true global restart could. `x0` itself is never perturbed away from, so a caller
that already has a good starting guess loses nothing.

Candidates are clamped into `[lower, upper]` before being passed to `f` rather than
handled via Optim.jl's `Fminbox` - plain `NelderMead` needs no gradient, and Fminbox's
log-barrier approach is built around methods that have one, so a barrier around a
derivative-free simplex search isn't a well-supported combination.
"""
function nelder_mead_optimize(f, x0::Array{Float64, 1}, options::Dict{Symbol, Any})
    n = length(x0)
    lower = get(options, :lower, 0.0)
    upper = get(options, :upper, 5000.0)
    lower_bounds = lower isa AbstractVector ? convert(Array{Float64, 1}, lower) : fill(convert(Float64, lower), n)
    upper_bounds = upper isa AbstractVector ? convert(Array{Float64, 1}, upper) : fill(convert(Float64, upper), n)
    box_range = upper_bounds .- lower_bounds

    maxfevals = get(options, :maxfevals, 15000)
    restarts = get(options, :restarts, 5)
    restart_scale = get(options, :restart_scale, 0.2)

    clamped_f = x -> f(clamp.(x, lower_bounds, upper_bounds))
    per_start_budget = max(1, maxfevals ÷ (restarts + 1))

    # iterations is capped at the same budget as f_calls_limit - Options' own
    # default (1_000) would otherwise silently cut a run short before
    # f_calls_limit binds, for any per_start_budget above that.
    nm_options = Optim.Options(f_calls_limit=per_start_budget, iterations=per_start_budget)

    best_x = clamp.(x0, lower_bounds, upper_bounds)
    best_f = clamped_f(best_x)

    result = Optim.optimize(clamped_f, best_x, Optim.NelderMead(), nm_options)
    if Optim.minimum(result) < best_f
        best_f = Optim.minimum(result)
        best_x = clamp.(Optim.minimizer(result), lower_bounds, upper_bounds)
    end

    for _ in 1:restarts
        start = clamp.(best_x .+ restart_scale .* box_range .* (2 .* rand(n) .- 1), lower_bounds, upper_bounds)
        result = Optim.optimize(clamped_f, start, Optim.NelderMead(), nm_options)
        candidate_f = Optim.minimum(result)
        if candidate_f < best_f
            best_f = candidate_f
            best_x = clamp.(Optim.minimizer(result), lower_bounds, upper_bounds)
        end
    end

    return best_x
end

function bboptimize(f, x0, params)
    start = Dates.now()
    latest = start

    best_f = f(x0)
    best_x = copy(x0)
    
    last_progress = 0

    # Standard differential-evolution guidance sizes the population to
    # several times the parameter count (~5-10x is typical) rather than a
    # flat constant - a fixed pool_size=6 was fine for newsvendor's 2
    # parameters (3x) but starved beer_game's 6-parameter search of the
    # diversity a DE/rand/1-style mutation (below) needs: its convergence
    # curve was still improving at the very end of its evaluation budget
    # (see benchmark/compare_optimizers.jl's beer_game results) - the
    # signature of a population too small to adequately cover the
    # landscape, not of stopping too early.
    pool_size = max(6, 5 * length(x0))
    candidate_pool = [rand(length(x0)) .* (params[:SearchRange][2] - params[:SearchRange][1]) .+ params[:SearchRange][1] for i in 1:pool_size]
    #println(candidate_pool)
    pool_f = [f(candidate) for candidate in candidate_pool]
    #println(pool_f)

    t = max(0.1, min(0.9, 6 / length(x0)))
    #println(t)

    # A bigger pool_size dilutes how often any specific slot gets refined
    # per raw evaluation - each mutation only touches one randomly-chosen
    # pool member, so a pool 5x the old flat size of 6 needs roughly 5x as
    # many evaluations to give every slot a comparable chance to improve.
    # Scaling MaxStepsWithoutProgress's default by the same pool_size/6
    # ratio that grew the population keeps the *fraction of a full
    # population cycle* the search is willing to wait roughly constant,
    # instead of the flat 10%-of-budget alone, which let a real search
    # quit - with zero improvement ever found - after fewer evaluations
    # than a single pass through a 30-member pool typically needs (see
    # test/policy-beergame-tests.jl's "Beer game" test, which caught this
    # exact case at pool_size=30).
    max_steps_without_progress = get(params, :MaxStepsWithoutProgress,
                                      max(100.0, 0.1 * params[:MaxFuncEvals] * (pool_size / 6.0)))

    for i in 1:params[:MaxFuncEvals]
        if i > last_progress + max_steps_without_progress
            println("$i, $(Dates.now() - start), $best_f, $best_x")
            break
        end

        i1 = rand(1:pool_size)
        i2 = rand(1:pool_size)
        i3 = rand(1:pool_size)

        candidate = copy(candidate_pool[i1])
        @inbounds for j in eachindex(candidate)
            r = rand()
            if r < 0.01
                candidate[j] = params[:SearchRange][1]
            elseif r < 0.02
                candidate[j] = candidate_pool[i1][j] + 2 * (randn())
            elseif r < 0.03
                k = rand(1:length(candidate))
                candidate[j] = candidate_pool[i1][k] + 1 * (randn())
                candidate[k] = candidate_pool[i1][j] + 1 * (randn())
            elseif r < 0.12
                candidate[j] = candidate_pool[i2][j] + 1 * (randn())
            elseif r < t + 0.12
                candidate[j] = candidate_pool[i1][j] + (rand() + 0.3) * (best_x[j] - candidate_pool[i3][j]) + (randn()) / 2
            end
            if candidate[j] < params[:SearchRange][1]
                candidate[j] = params[:SearchRange][1] + rand()^3 * (params[:SearchRange][2] - params[:SearchRange][1])
            end
            if candidate[j] > params[:SearchRange][2]
                candidate[j] = params[:SearchRange][2] - rand()^3 * (params[:SearchRange][2] - params[:SearchRange][1])
            end
        end
        candidate_f = f(candidate)
        #println("$i, $(Dates.now() - start), $candidate_f, $candidate")
        if (candidate_f ≈ pool_f[i1]) && (sum(candidate_f) < sum(pool_f[i1]))
            pool_f[i1] = candidate_f
            candidate_pool[i1] = candidate
        end
        if candidate_f < pool_f[i1]
            pool_f[i1] = candidate_f
            candidate_pool[i1] = candidate
        end
        if (candidate_f ≈ best_f) && (sum(candidate_f) < sum(best_f))
            best_f = candidate_f
            best_x = copy(candidate)
            println("*- $i, $(Dates.now() - start), $best_f, $best_x")
        end
        if candidate_f < best_f
            best_f = candidate_f
            best_x = copy(candidate)
            last_progress = i
            println("** $i, $(Dates.now() - start), $best_f, $best_x")
        end

        if i % 200 == 0
            println("$i, $(Dates.now() - start), $(Dates.now() - latest), $best_f")#, $best_x")
            latest = Dates.now()
        end
    end
    return best_x
end

#function minimizer(results::BlackBoxOptim.OptimizationResults)
#    return best_candidate(results)
#end

function minimizer(x)
    return x
end