abstract type Transport end

struct Lane
    origin
    destination

    unit_cost

    lead_time

    function Lane(;origin=origin, destination=destination, unit_cost=unit_cost)
        return new(origin, destination, unit_cost, 0)
    end
end

struct Truck
    capacities
    
    fixed_cost
    unit_cost
end

struct Route
    truck

    origin
    destinations::Array{L1, 1} where L1 <: Location

    departures

    lead_times::Dict{L2, Int64} where L2 <: Location
end

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
