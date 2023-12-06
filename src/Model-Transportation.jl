using UUIDs

import Base.isequal

struct Truck
    capacities
    
    fixed_cost
end

"""
A trip is the basis of transportation in the simulation. It follows a route with a given departure time.
"""
struct Trip
    route::Lane
    departure::Int64
    policies::Union{Missing, Dict{Product, <:InventoryOrderingPolicy}}
end

function get_trips(lane::Lane, horizon::Int64)
    return [Trip(lane, t, missing) for t in 1:horizon if (isnothing(lane.can_ship) || isempty(lane.can_ship) || lane.can_ship[t])]
end

function get_trips(lanes::Array{Lane, 1}, horizon::Int64)
    return [Trip(l, t, missing) for l in lanes for t in 1:horizon if (isnothing(l.can_ship) || isempty(l.can_ship) || l.can_ship[t])]
end

function get_trips(lanes::Set{Lane}, horizon::Int64)
    return [Trip(l, t, missing) for l in lanes for t in 1:horizon if (isnothing(l.can_ship) || isempty(l.can_ship) || l.can_ship[t])]
end

function get_trips(routes, horizon::Int64)
    return [Trip(r, t, missing) for r in routes for t in 1:horizon]
end

function get_trips(supplychain, policies)
    return [Trip(l, 
                 t, 
                 Dict(collect(p => policies[(l, p)] for p in supplychain.products if haskey(policies, (l, p))))
                 ) for l in supplychain.lanes for t in 1:supplychain.horizon if (isnothing(l.can_ship) || isempty(l.can_ship) || l.can_ship[t])]
end