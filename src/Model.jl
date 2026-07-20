include("Model-Transportation.jl")

"""
Not parametrized on origin/destination type (unlike a first attempt at
this struct): OrderLine collections are inherently heterogeneous (a single
Vector{OrderLine}/Set{OrderLine} holds order lines between every pairing
of node types in the network), so a caller constructing one from an
abstractly-typed value (e.g. `trip.route.origin::ConcreteNode`) can never
supply a statically-known concrete O/D anyway - parametrizing just left
the type parameter unresolved (`OrderLine{O,D} where O<:Node`), which is
worse than a plain abstract field: every downstream use (push!, Set
membership, dispatch) then has to deal with an unresolved existential type
instead of a single well-known concrete struct.
"""
mutable struct OrderLine
    creation_time::Int64
    origin::ConcreteNode # from
    destination::ConcreteNode # to
    product::Product
    quantity::Int64
    due_date::Int64 # when

    trip::Union{Missing, Trip} # how (filled when shipping)
end

function get_inbound_trips(env, location, time)
    return env.departures[location][time]
end

"""
    find_next_departure(env, destination, time)

Finds the earliest trip bound for `destination` that departs at or after
`time`, or `nothing` if none remain within the horizon. Walks forward through
`env.departures[destination]` (indexed directly by period) instead of
filtering the full, unbounded list of trips ever bound for `destination`.
"""
function find_next_departure(env, destination::ConcreteNode, time::Int64)
    periods = env.departures[destination]
    for t in time:length(periods)
        trips_at_t = periods[t]
        if !isempty(trips_at_t)
            return trips_at_t[1]
        end
    end
    return nothing
end

"""
    find_next_departure(env, destination, time, due_date)

Finds the earliest trip bound for `destination` that departs at or after
`time` and still arrives by `due_date`, or `nothing` if none do. Only scans
the `[time, due_date]` window of `env.departures[destination]`, instead of
the full, unbounded list of trips ever bound for `destination`.
"""
function find_next_departure(env, destination::ConcreteNode, time::Int64, due_date::Int64)
    periods = env.departures[destination]
    last_period = min(due_date, length(periods))
    for t in time:last_period
        for trip in periods[t]
            if t + trip.route.times[1] <= due_date
                return trip
            end
        end
    end
    return nothing
end

"""
    get_locations(supplychain)

    Gets all the locations (storages, customers, and suppliers - not
    plants) in the supplychain, as the same cached Vector every call (see
    `get_location_index` in SupplyChainModeling.jl).
"""
function get_locations(supplychain::SupplyChain)
    return get_location_index(supplychain).items
end

function create_graph(supplychain::SupplyChain)
    graph = Graphs.DiGraph(length(get_locations(supplychain)))

    mapping = Dict{ConcreteNode, Int64}()
    i = 1
    for location in get_locations(supplychain)
        mapping[location] = i
        i += 1
    end

    # Every Trip's route is just the Lane it was built from (see
    # Model-Transportation.jl), and lanes are already unique - materializing
    # a Trip per (lane, period) via get_trips only to immediately discard
    # everything but trip.route was a redundant O(lanes * horizon) pass.
    for route in supplychain.lanes
        for destination in get_destinations(route)
            Graphs.add_edge!(graph, mapping[route.origin], mapping[destination])
        end
    end

    return (graph, mapping)
end

function get_sorted_locations(supplychain)::Vector{ConcreteNode}
    (graph, mapping) = create_graph(supplychain)

    reverse_mapping = Vector{eltype(mapping.keys)}(undef, length(mapping))
    for (k, v) in mapping
        reverse_mapping[v] = k
    end

    return reverse_mapping[topological_sort_by_dfs(graph)]
end

function get_downstream_customers(supplychain, location)
    (graph, mapping) = create_graph(supplychain)

    reverse_mapping = Vector{eltype(mapping.keys)}(undef, length(mapping))
    for (k, v) in mapping
        reverse_mapping[v] = k
    end

    parents = dfs_parents(graph, mapping[location])

    return filter(n -> isa(n, Customer), map(i -> reverse_mapping[i], filter(i -> parents[i] > 0, 1:length(get_locations(supplychain)))))
end