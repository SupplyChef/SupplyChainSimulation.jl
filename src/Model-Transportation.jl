using UUIDs

import Base.isequal


abstract type Transport end

struct Lane <: Transport
    origin::Union{Supplier, Storage, Customer}
    destination::Union{Supplier, Storage, Customer}

    fixed_cost::Float64
    unit_cost::Float64

    lead_time::Int64

    can_ship::Array{Bool, 1}

    function Lane(;origin, destination, fixed_cost=0, unit_cost=0, lead_time=0, can_ship::Array{Bool, 1}=Bool[])
        return new(origin, destination, fixed_cost, unit_cost, lead_time, can_ship)
    end
end

Base.:(==)(x::Lane, y::Lane) = x.origin == y.origin &&  x.destination == y.destination
Base.hash(x::Lane, h::UInt64) = hash(x.origin, hash(x.destination, h))
Base.show(io::IO, x::Lane) = print(io, "$(x.origin) $(x.destination)")

struct Truck
    capacities
    
    fixed_cost
end

struct Route <: Transport
    id
    truck::Truck

    origin
    destinations::Array{L1, 1} where L1 <: Location

    departures

    lead_times::Dict{L2, Int64} where L2 <: Location

    unit_cost

    function Route(;origin, destinations, fixed_cost=0, unit_cost=0)
        return new(UUIDs.uuid1(), Truck(0, fixed_cost), origin, destinations, [], Dict(d => 0 for d in destinations), unit_cost)
    end
end

Base.hash(r::Route, h::UInt) = hash(r.id, h)
Base.isequal(r1::Route, r2::Route) = isequal(r1.id, r2.id)

"""
A trip is the basis of transportation in the simulation. It follows a route with a given departure time.
"""
struct Trip
    route

    departure::Int64
end

function get_destinations(route::Route)
    return route.destinations
end

function get_destinations(lane::Lane)
    return [lane.destination]
end

function is_destination(location, route::Route)
    return location ∈ get_destinations(route)
end

function is_destination(location, lane::Lane)
    return location == lane.destination
end

function get_leadtime(route::Route, destination)
    return route.lead_times[destination]
end

function get_leadtime(lane::Lane, destination)
    return lane.lead_time
end

function get_trips(route::Route, horizon)
    return [Trip(route, t) for t in 1:horizon]
end

function get_trips(lane::Lane, horizon)
    return [Trip(lane, t) for t in 1:horizon if (isempty(lane.can_ship) || lane.can_ship[t])]
end

function get_trips(lanes::Array{Lane, 1}, horizon)
    return [Trip(l, t) for l in lanes for t in 1:horizon if (isempty(l.can_ship) || l.can_ship[t])]
end

function get_trips(routes, horizon)
    return [Trip(r, t) for r in routes for t in 1:horizon]
end

function get_fixed_cost(lane::Lane)
    return lane.fixed_cost
end

function get_fixed_cost(route::Route)
    return route.truck.fixed_cost
end