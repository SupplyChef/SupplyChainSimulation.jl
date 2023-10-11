
function minimizer(results::BlackBoxOptim.OptimizationResults)
    return best_candidate(results)
end

function minimize!(env::Env, horizon::Int64, initial_states::Array{State, 1}, policies::Array{P, 1}, x::Array{Float64, 1}; cost_function=s->get_total_lost_sales(s) + 0.001 * get_total_orders(s)) where P <: InventoryOrderingPolicy
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
function optimize!(network::Network, horizon::Int64, initial_states...; cost_function=s->get_total_lost_sales(s) + get_total_holding_costs(s) + get_total_trip_fixed_costs(s))
    env = Env(network, initial_states)

    initial_state = initial_states[1]
    policies = unique([initial_state.policies[k] for k in keys(initial_state.policies)])
    println(policies)

    parameter_count = sum(get_parameter_count(policy) for policy in policies)
    println(parameter_count)
    x0 = zeros(Float64, parameter_count)
    res = bboptimize(x -> minimize!(env, horizon, collect(initial_states), policies, x; cost_function=cost_function), 
                     x0, 
                    Dict(:MaxFuncEvals => 3000,
                         :MaxStepsWithoutProgress => 100, 
                         :SearchRange => (-0.0, 5000.0), 
                         :NumDimensions => parameter_count, 
                         :Method => :generating_set_search))

    best = minimizer(res)
    i = 1
    for policy in policies
        set_parameter!(policy, best[i:i+get_parameter_count(policy)-1])
        i = i + get_parameter_count(policy)
    end
end

function optimize(f, x0; SearchRange, NumDimensions, MaxFuncEvals, Method)

end
