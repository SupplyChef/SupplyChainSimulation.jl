using Memoize

struct Env 
    network
    initial_states

    supplying_trips

    function Env(network, initial_states)
        return new(network, initial_states, Dict(location => get_inbound_trips(network, location) for location in get_locations(network)))
    end
end

function get_inbound_trips(network, location)
    return collect(filter(trip -> is_destination(location, trip.route), network.trips))
end

function get_mean_demands(env::Env)

end

@memoize Dict function get_mean_demand(env::Env, customer::Customer, product::Product, time::Int)
    demand = sum(initial_state.demand[(customer, product)][time] for initial_state in env.initial_states) / length(env.initial_states)
    return demand
end

@memoize Dict function get_mean_demand(env::Env, location::Location, product::Product, time::Int)
    demand = 0
    for customer in get_downstream_customers(env.network, location)
        demand = demand + sum(initial_state.demand[(customer, product)][time] for initial_state in env.initial_states) / length(env.initial_states)
    end
    return demand
end