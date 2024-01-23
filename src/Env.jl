"""
Contains information about the environment of the simulation, including the network configuration.
"""
struct Env 
    supplychain::SupplyChain
    initial_states::Array{State, 1}
    
    sorted_locations::Array{<:Node, 1}
    sorted_products::Array{Product, 1}
    supplying_trips::Dict{Node, Array{Trip, 1}}

    function Env(supplychain::SupplyChain, initial_states, policies)
        trips = get_trips(supplychain, policies)
        sorted_locations = get_sorted_locations(supplychain)

        return new(supplychain, 
                   collect(initial_states),
                   sorted_locations, 
                   collect(supplychain.products),
                   Dict(location => sort(collect(filter(trip -> is_destination(location, trip.route), trips)), 
                                         by=t->t.departure) for location in get_locations(supplychain)))
    end
end

function get_inbound_trips(supplychain, location)
    return collect(sort(filter(trip -> is_destination(location, trip.route), get_trips(supplychain.lanes, supplychain.horizon)), by=t -> t.unit_cost))
end

function get_mean_demands(env::Env)

end

function get_mean_demand(env::Env, customer::Customer, product::Product, time::Int)
    demand = sum(initial_state.demand[(customer, product)].demand[time] for initial_state in env.initial_states) / length(env.initial_states)
    return demand
end

function get_mean_demand(env::Env, location::Node, product::Product, time::Int)
    demand = 0
    for customer in get_downstream_customers(env.supplychain, location)
        demand = demand + sum(initial_state.demand[(customer, product)].demand[time] for initial_state in env.initial_states) / length(env.initial_states)
    end
    return demand
end