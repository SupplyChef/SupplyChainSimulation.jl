
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
"""
function optimize!(lane_policies, supplychains...; params::Dict{Symbol, Float64}=Dict{Symbol, Float64}(), cost_function=s->-get_total_sales(s) + get_total_lost_sales(s) + get_total_holding_costs(s) + get_total_trip_fixed_costs(s) + get_total_trip_unit_costs(s) + 0.001 * get_total_orders(s), record_history::Bool=true, method::Symbol=:custom, cma_es_options::Dict{Symbol, Any}=Dict{Symbol, Any}())
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
        best = SupplyChainSimulation.bboptimize(f,
                         x0,
                        merge(Dict(:MaxFuncEvals => 15000,
                             :MaxStepsWithoutProgress => 1500,
                             :SearchRange => (-0.0, 5000.0),
                             :NumDimensions => length(x0)), params))
    elseif method === :cma_es
        best = cma_es_optimize(f, x0, cma_es_options)
    else
        error("optimize!: unknown method $(repr(method)); supported methods are :custom and :cma_es")
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

function bboptimize(f, x0, params)
    start = Dates.now()
    latest = start

    best_f = f(x0)
    best_x = copy(x0)
    
    last_progress = 0
    
    pool_size = 6
    candidate_pool = [rand(length(x0)) .* (params[:SearchRange][2] - params[:SearchRange][1]) .+ params[:SearchRange][1] for i in 1:pool_size]
    #println(candidate_pool)
    pool_f = [f(candidate) for candidate in candidate_pool]
    #println(pool_f)

    t = max(0.1, min(0.9, 6 / length(x0)))
    #println(t)

    for i in 1:params[:MaxFuncEvals]
        if i > last_progress + params[:MaxStepsWithoutProgress]
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