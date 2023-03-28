
function minimizer(results::BlackBoxOptim.OptimizationResults)
    return best_candidate(results)
end

function minimize!(network::Network, horizon::Int64, initial_states, policies, x; cost_function=s->get_total_lost_sales(s) + 0.001 * get_total_orders(s))
    i = 1
    for policy in policies
        set_parameter!(policy, x[i:i+get_parameter_count(policy)-1])
        i = i + get_parameter_count(policy)
    end

    value = 0
    for initial_state in initial_states
        #println(initial_state)
        final_state = simulate(network, horizon, initial_state)

        #println(final_state)

        #println("$x $(get_total_lost_sales(final_state)) $(get_total_orders(final_state))")
        value += cost_function(final_state)
    end
    return value
end

function optimize!(network::Network, horizon::Int64, initial_states...)
    
    initial_state = initial_states[1]
    policies = [initial_state.policies[k] for k in keys(initial_state.policies)]

    parameter_count = sum(get_parameter_count(policy) for policy in policies)
    x0 = zeros(Float64, parameter_count)
    res = bboptimize(x -> minimize!(network, horizon, initial_states, policies, x), x0; 
                SearchRange=(-0.0, 50.0), NumDimensions=parameter_count, Method = :de_rand_1_bin)

    best = minimizer(res)
    i = 1
    for policy in policies
        set_parameter!(policy, best[i:i+get_parameter_count(policy)-1])
        i = i + get_parameter_count(policy)
    end
end
