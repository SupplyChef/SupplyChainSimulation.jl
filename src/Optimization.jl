
using Distributions
using Dates

function minimize!(env::Env, lane_policies, policies, initial_states::Array{State, 1}, x::Array{Float64, 1}; cost_function)
    i = 1
    for policy in policies
        set_parameters!(policy, x[i:i+length(get_parameters(policy))-1])
        i = i + length(get_parameters(policy))
    end

    value = 0
    for initial_state in initial_states
        #println(initial_state)
        final_state = simulate(env, lane_policies, initial_state)

        #println(final_state)

        #println("$x $(get_total_lost_sales(final_state)) $(get_total_orders(final_state))")
        value += cost_function(final_state)
    end
    return value
end

"""
    optimize!(supplychain::SupplyChain, lane_policies, initial_states...; cost_function)

    Optimizes the inventory policies in the supply chain by simulating the inventory movement starting from the initial states and costing the results with the cost function.
"""
function optimize!(supplychain::SupplyChain, lane_policies::Dict{Tuple{Lane, Product}, <:InventoryOrderingPolicy}, initial_states...; params::Dict{Symbol, Float64}=Dict{Symbol, Float64}(), cost_function=s->get_total_lost_sales(s) + get_total_holding_costs(s) + get_total_trip_fixed_costs(s) + get_total_trip_unit_costs(s) + 0.001 * get_total_orders(s))
    env = Env(supplychain, initial_states, lane_policies)

    policies = unique([lane_policies[k] for k in keys(lane_policies)])
    #println(policies)

    x0 = vcat([get_parameters(policy) for policy in policies]...)
    x0 = convert(Array{Float64, 1}, x0)
    println(x0)
    res1 = SupplyChainSimulation.bboptimize(x -> minimize!(env, lane_policies, policies, collect(initial_states), x; cost_function=cost_function), 
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
    
    t = max(0.1, min(0.9, 6 / length(x0)))
    println(t)

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
        #println("$candidate_f, $candidate")
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

        if i % 50 == 0
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