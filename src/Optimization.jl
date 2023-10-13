
using Distributions
using Dates

function minimize!(env::Env, horizon::Int64, initial_states::Array{State, 1}, policies::Array{P, 1}, x::Array{Float64, 1}; cost_function) where P <: InventoryOrderingPolicy
    i = 1
    for policy in policies
        set_parameter!(policy, x[i:i+get_parameter_count(policy)-1])
        i = i + get_parameter_count(policy)
    end

    value = 0
    for initial_state in initial_states
        #println(initial_state)
        final_state = simulate(env, horizon, initial_state)

        #println(final_state)

        #println("$x $(get_total_lost_sales(final_state)) $(get_total_orders(final_state))")
        value += cost_function(final_state)
    end
    return value
end

"""
    optimize!(network::Network, horizon::Int64, initial_states...; cost_function)

    Optimizes the inventory policies in the network by simulating the inventory movement starting from the initial states and costing the results with the cost function.
"""
function optimize!(network::Network, horizon::Int64, initial_states...; cost_function=s->get_total_lost_sales(s) + get_total_holding_costs(s) + get_total_trip_fixed_costs(s) + 0.001 * get_total_orders(s))
    env = Env(network, initial_states)

    initial_state = initial_states[1]
    policies = unique([initial_state.policies[k] for k in keys(initial_state.policies)])
    println(policies)

    parameter_count = sum(get_parameter_count(policy) for policy in policies)
    println(parameter_count)
    x0 = zeros(Float64, parameter_count)
    res1 = SupplyChainSimulation.bboptimize(x -> minimize!(env, horizon, collect(initial_states), policies, x; cost_function=cost_function), 
                     x0, 
                    Dict(:MaxFuncEvals => 15000,
                         :MaxStepsWithoutProgress => 1500, 
                         :SearchRange => (-0.0, 5000.0), 
                         :NumDimensions => parameter_count))

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
        set_parameter!(policy, best[i:i+get_parameter_count(policy)-1])
        i = i + get_parameter_count(policy)
    end
end

function bboptimize(f, x0, params)
    start = Dates.now()

    best_f = f(x0)
    best_x = copy(x0)
    
    last_progress = 0
    
    pool_size = 12
    candidate_pool = [rand(length(x0)) .* (params[:SearchRange][2] - params[:SearchRange][1]) .+ params[:SearchRange][1] for i in 1:pool_size]
    #println(candidate_pool)
    pool_f = [f(candidate) for candidate in candidate_pool]
    
    t = max(0.1, min(0.9, 4 / length(x0)))
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
        @inbounds for j in 1:length(candidate)
            r = rand()
            if r < 0.01
                candidate[j] = params[:SearchRange][1]
            elseif r < 0.02
                candidate[j] = candidate_pool[i1][j] + 2 * (rand() - 0.5)
            elseif r < 0.12
                candidate[j] = candidate_pool[i2][j] + 2 * (rand() - 0.5)
            elseif r < t + 0.12
                candidate[j] = candidate_pool[i1][j] + rand() * (best_x[j] - candidate_pool[i3][j]) + (rand() - 0.5) / 10
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

        if i % 500 == 0
            println("$i, $(Dates.now() - start), $best_f, $best_x")
        end
    end
    return best_x
end

function minimizer(results::BlackBoxOptim.OptimizationResults)
    return best_candidate(results)
end

function minimizer(x)
    return x
end