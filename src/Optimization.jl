
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

function minimize!(lane_policies, policies, envs::Array{Env, 1}, initial_states::Array{State, 1}, x::Array{Float64, 1}; cost_function)
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
    optimize!(supplychain::SupplyChain, lane_policies, initial_states...; cost_function, record_history)

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

    `customer_backlog` (default `false`, matching every trial simulation's
    behavior before this field existed) is passed straight through to `Env` -
    see its docstring there for what it changes.
"""
function optimize!(lane_policies, supplychains...; params::Dict{Symbol, Float64}=Dict{Symbol, Float64}(), cost_function=s->-get_total_sales(s) + get_total_lost_sales(s) + get_total_holding_costs(s) + get_total_trip_fixed_costs(s) + get_total_trip_unit_costs(s) + 0.001 * get_total_orders(s), record_history::Bool=true, customer_backlog::Bool=false)
    initial_states = State.(supplychains)
    envs = [Env(supplychain, initial_states, lane_policies; record_history=record_history, customer_backlog=customer_backlog) for supplychain in supplychains]

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
    
    res1 = SupplyChainSimulation.bboptimize(x -> minimize!(lane_policies, policies, collect(envs), collect(initial_states), x; cost_function=cost_function), 
                     x0, 
                    merge(Dict(:MaxFuncEvals => 15000,
                         :MaxStepsWithoutProgress => 1500, 
                         :SearchRange => (-0.0, 5000.0), 
                         :NumDimensions => length(x0)), params))

    # res = BlackBoxOptim.bboptimize(x -> minimize!(env, horizon, collect(initial_states), policies, x; cost_function=cost_function), 
    #                      x0, 
    #                     Dict(:MaxFuncEvals => 3000,
    #                          :MaxStepsWithoutProgress => 500, 
    #                          :SearchRange => (-0.0, 5000.0), 
    #                          :NumDimensions => parameter_count, 
    #                          :Method => :generating_set_search,
    #                          :TraceMode => :silent))

    #best = minimizer(res)
    #println(best)
    #println(best_fitness(res))
    best = res1
    i = 1
    for policy in policies
        set_parameters!(policy, best[i:i+length(get_parameters(policy))-1])
        i = i + length(get_parameters(policy))
    end
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