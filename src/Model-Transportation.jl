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
    policies::Union{Missing, Dict{Product, InventoryOrderingPolicy}}
end

# Without these, Set{Trip}/Dict{Trip,...} (metrics.seen_trips,
# state.historical_transportation) fall back to Julia's default
# identity-based (objectid) hashing/equality for structs with non-isbits
# fields (route::Lane and policies::Dict aren't isbits) - CPU profiling
# optimize!'s hot loop found this as the single largest self-time cost,
# since record_fill! checks/inserts into metrics.seen_trips on every order
# line filled. policies isn't part of identity here: every Trip built from
# the same (lane, departure) shares the same policies dict reference (see
# get_lane_policies), so (route, departure) alone is exactly the same
# equivalence classes identity-based comparison already produced - this is
# a hashing speedup, not a semantic change.
Base.:(==)(x::Trip, y::Trip) = x.route == y.route && x.departure == y.departure
Base.hash(x::Trip, h::UInt64) = hash(x.departure, hash(x.route, h))

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

Builds one `Product => policy` dict per lane. The per-period `policies`
dict attached to each `Trip` only ever depends on the lane, not the period,
so sharing a single dict across every period of a lane (instead of
rebuilding an identical one per (lane, period) pair) avoids `horizon`-many
redundant Dict allocations per lane.

Stores each policy object directly, rather than pre-wrapping it in a
`FunctionWrapper` closure via `wrap_get_order` - that type erasure went
through a `ccall`-based trampoline, which forced boxing of every argument
crossing the boundary (`Storage`, `Lane`, `Env`, `Product`). `place_orders`
(Simulation.jl) hand-writes a union split (an explicit `isa` chain, each
branch followed by a type assertion) inline at its `get_order` call site,
rather than calling `get_order(policy, ...)` directly on the abstractly-
typed `policy` it reads out of this dict: `Storage`, `Lane`, `Env`, and
`Product` are immutable (non-`isbits`) structs, so a plain call on an
abstractly-typed `policy` still boxes them the same way FunctionWrapper
did - the boxing came from the abstract-typed dynamic dispatch itself, not
from the ccall trampoline specifically. The union split has to be written
inline in `place_orders`, not factored into a helper function: Julia's
`isa`+type-assert narrowing is local to the function scope it happens in,
so calling out to a helper with `policy` still abstractly typed
re-introduces the exact same dynamic dispatch (confirmed the hard way -
factoring it into a `dispatch_get_order` helper measured identically to no
split at all, since the call *into* that helper was itself still dynamic).
"""
function get_lane_policies(supplychain, policies)
    return Dict{Lane, Dict{Product, InventoryOrderingPolicy}}(
        l => Dict{Product, InventoryOrderingPolicy}(p => policies[(l, p)] for p in supplychain.products if haskey(policies, (l, p)))
        for l in supplychain.lanes
    )
end

function get_trips(supplychain, policies)
    lane_policies = get_lane_policies(supplychain, policies)
    return [Trip(l, t, lane_policies[l]) for l in supplychain.lanes for t in 1:supplychain.horizon if (isnothing(l.can_ship) || isempty(l.can_ship) || l.can_ship[t])]
end

const NULL_LANE = Lane(Customer("NULL"), Customer("NULL"); unit_cost=0.0)
const NULL_TRIP = Trip(NULL_LANE, 0, missing)