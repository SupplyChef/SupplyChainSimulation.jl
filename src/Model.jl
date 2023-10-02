abstract type Location end

struct Supplier <: Location
    name
end

Base.:(==)(x::Supplier, y::Supplier) = x.name == y.name 
Base.hash(x::Supplier, h::UInt64) = hash(x.name, h)
Base.show(io::IO, x::Supplier) = print(io, x.name)

struct Storage <: Location 
    name
end

Base.:(==)(x::Storage, y::Storage) = x.name == y.name 
Base.hash(x::Storage, h::UInt64) = hash(x.name, h)
Base.show(io::IO, x::Storage) = print(io, x.name)

struct Customer <: Location 
    name
end

Base.:(==)(x::Customer, y::Customer) = x.name == y.name 
Base.hash(x::Customer, h::UInt64) = hash(x.name, h)
Base.show(io::IO, x::Customer) = print(io, x.name)

abstract type Product end

struct Single <: Product
    name
    holding_costs

    function Single(name, holding_costs=0.0)
        return new(name, holding_costs)
    end
end

Base.:(==)(x::Product, y::Product) = x.name == y.name 
Base.hash(x::Product, h::UInt64) = hash(x.name, h)
Base.show(io::IO, x::Product) = print(io, x.name)

struct Bundle <: Product
    name
    holding_costs

    composition::Dict{P, Float64} where P <: Product
end

struct OrderLine
    order
    product
    quantity
end

include("Model-Transportation.jl")

struct Order
    destination # where 
    trip # how
    lines::Array{OrderLine, 1} # what 
    due_date::Int64 # when

    function Order(destination, trip::Trip, lines::Array{OrderLine, 1}, due_date::Int64)
        return new(destination, trip, lines, due_date)
    end

    function Order(destination, trip::Trip, lines::Array{Tuple{P, Int64}, 1}, due_date::Int64) where P <: Product
        order = new(destination, trip, OrderLine[], due_date)
        for (product, quantity) in lines
            push!(order.lines, OrderLine(order, product, quantity))
        end
        return order
    end

    function Order(lane::Lane, lines::Array{Tuple{P, Int64}, 1}, due_date::Int64) where P <: Product
        order = new(lane.destination, Trip(lane, 1), OrderLine[], due_date)
        for (product, quantity) in lines
            push!(order.lines, OrderLine(order, product, quantity))
        end
        return order
    end
end

struct Network 
    suppliers::Array{Supplier, 1}
    storages::Array{Storage, 1}
    customers::Array{Customer, 1}
    
    trips::Array{Trip, 1}

    products::Array{P, 1}  where P <: Product
end

function get_inbound_trips(env, location, time)
    return collect(filter(trip -> trip.departure == time && is_destination(location, trip.route), env.supplying_trips[location]))
end

"""
    get_locations(network)

    Gets all the locations in the network.
"""
function get_locations(network)
    return vcat(network.storages, network.customers, network.suppliers)
end

function create_graph(network)
    graph = Graphs.DiGraph(length(get_locations(network)))

    mapping = Dict{Location, Int64}()
    i = 1
    for location in get_locations(network)
        mapping[location] = i
        i += 1
    end

    for route in unique(map(trip -> trip.route, network.trips))
        for destination in get_destinations(route)
            Graphs.add_edge!(graph, mapping[route.origin], mapping[destination])
        end
    end

    return (graph, mapping)
end

function get_sorted_locations(network)
    (graph, mapping) = create_graph(network)

    reverse_mapping = Vector{eltype(mapping.keys)}(undef, length(mapping))
    for (k, v) in mapping
        reverse_mapping[v] = k
    end

    return reverse_mapping[topological_sort_by_dfs(graph)]
end

function get_downstream_customers(network, location)
    (graph, mapping) = create_graph(network)

    reverse_mapping = Vector{eltype(mapping.keys)}(undef, length(mapping))
    for (k, v) in mapping
        reverse_mapping[v] = k
    end

    parents = dfs_parents(graph, mapping[location])

    return filter(n -> isa(n, Customer), map(i -> reverse_mapping[i], filter(i -> parents[i] > 0, 1:length(get_locations(network)))))
end