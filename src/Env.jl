"""
Contains information about the environment of the simulation, including the network configuration.
"""
struct Env 
    supplychain::SupplyChain
    initial_states
    
    supplying_trips::Dict{Node, Array{Trip, 1}}

    function Env(supplychain, initial_states, policies)
        trips = get_trips(supplychain, policies)
        return new(supplychain, 
                   initial_states, 
                   Dict(location => sort(collect(filter(trip -> is_destination(location, trip.route), trips)), 
                                         by=t->t.departure) for location in get_locations(supplychain)))
    end
end

function get_inbound_trips(supplychain, location)
    return collect(filter(trip -> is_destination(location, trip.route), get_trips(supplychain.lanes, supplychain.horizon)))
end

function get_mean_demands(env::Env)

end

function get_mean_demand(env::Env, customer::Customer, product::Product, time::Int)
    demand = sum(initial_state.demand[(customer, product)][time] for initial_state in env.initial_states) / length(env.initial_states)
    return demand
end

function get_mean_demand(env::Env, location::Node, product::Product, time::Int)
    demand = 0
    for customer in get_downstream_customers(env.supplychain, location)
        demand = demand + sum(initial_state.demand[(customer, product)][time] for initial_state in env.initial_states) / length(env.initial_states)
    end
    return demand
end