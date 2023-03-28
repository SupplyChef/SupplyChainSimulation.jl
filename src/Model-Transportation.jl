using UUIDs

import Base.isequal


abstract type Transport end

struct Lane <: Transport
    origin
    destination

    unit_cost

    lead_time

    function Lane(;origin, destination, unit_cost=0)
        return new(origin, destination, unit_cost, 0)
    end
end

struct Truck
    capacities
    
    fixed_cost
    unit_cost
end

struct Route <: Transport
    id
    truck::Truck

    origin
    destinations::Array{L1, 1} where L1 <: Location

    departures

    lead_times::Dict{L2, Int64} where L2 <: Location

    function Route(;origin, destinations, fixed_cost=0, unit_cost=0)
        return new(UUIDs.uuid1(), Truck(0, fixed_cost, unit_cost), origin, destinations, [], Dict(d => 0 for d in destinations))
    end
end

Base.hash(r::Route, h::UInt) = hash(r.id, h)
Base.isequal(r1::Route, r2::Route) = isequal(r1.id, r2.id)

struct Trip
    route

    departure
end

function get_destinations(route::Route)
    return route.destinations
end

function get_destinations(lane::Lane)
    return [lane.destination]
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
    return [Trip(lane, t) for t in 1:horizon]
end

function get_trips(lanes::Array{Lane, 1}, horizon)
    return [Trip(l, t) for l in lanes for t in 1:horizon]
end

function get_trips(routes, horizon)
    return [Trip(r, t) for r in routes for t in 1:horizon]
end