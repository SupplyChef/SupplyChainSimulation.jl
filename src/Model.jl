include("Model-Transportation.jl")

mutable struct OrderLine{O<:Node, D<:Node}
    creation_time::Int64
    origin::O # from
    destination::D # to 
    product::Product
    quantity::Int64
    due_date::Int64 # when

    trip::Union{Missing, Trip} # how (filled when shipping)
    
    #function OrderLine{O, D}( creation_time::Int64,
    #                    origin::O, # from
    #                    destination::D, # to 
    #                    product::Product,
    #                    quantity::Int64,
    #                    due_date::Int64 # when
    #        ) where {O<:Node, D<:Node}
    #    return new(creation_time, origin, destination, product, quantity, due_date, missing)
    #end
end

function get_inbound_trips(env, location, time)
    return Iterators.filter(trip -> trip.departure == time, env.supplying_trips[location])
end

"""
    get_locations(supplychain)

    Gets all the locations in the supplychain.
"""
function get_locations(supplychain::SupplyChain)
    return union(supplychain.storages, supplychain.customers, supplychain.suppliers)
end

function create_graph(supplychain::SupplyChain)
    graph = Graphs.DiGraph(length(get_locations(supplychain)))

    mapping = Dict{Node, Int64}()
    i = 1
    for location in get_locations(supplychain)
        mapping[location] = i
        i += 1
    end

    for route in unique(map(trip -> trip.route, collect(get_trips(supplychain.lanes, supplychain.horizon))))
        for destination in get_destinations(route)
            Graphs.add_edge!(graph, mapping[route.origin], mapping[destination])
        end
    end

    return (graph, mapping)
end

function get_sorted_locations(supplychain)::Vector{<:Node}
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