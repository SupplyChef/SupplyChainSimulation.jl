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
    policies::Union{Missing, Dict{Product, GetOrderFn}}
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

"""
    get_lane_policies(supplychain, policies)

Builds one `Product => get_order-wrapper` dict per lane. The per-period
`policies` dict attached to each `Trip` only ever depends on the lane, not
the period, so sharing a single dict across every period of a lane (instead
of rebuilding an identical one per (lane, period) pair) avoids
`horizon`-many redundant Dict allocations per lane.

Wraps each policy via `wrap_get_order` (see Policy.jl) rather than storing
it directly: this resolves which concrete `get_order` method to call once,
here, instead of on every call in simulate()'s hot loop - see `GetOrderFn`.
"""
function get_lane_policies(supplychain, policies)
    return Dict{Lane, Dict{Product, GetOrderFn}}(
        l => Dict(p => wrap_get_order(policies[(l, p)]) for p in supplychain.products if haskey(policies, (l, p)))
        for l in supplychain.lanes
    )
end

function get_trips(supplychain, policies)
    lane_policies = get_lane_policies(supplychain, policies)
    return [Trip(l, t, lane_policies[l]) for l in supplychain.lanes for t in 1:supplychain.horizon if (isnothing(l.can_ship) || isempty(l.can_ship) || l.can_ship[t])]
end