"""
Contains information about the environment of the simulation, including the network configuration.
"""
struct Env
    supplychain::SupplyChain
    initial_states::Array{State, 1}

    # Concretely Vector{Node} (not `Array{<:Node, 1}`, a UnionAll): the latter
    # made the field itself abstractly typed, forcing dynamic dispatch on
    # every access even though the stored vector's element type never
    # changes for a given Env.
    sorted_locations::Vector{Node}
    sorted_products::Array{Product, 1}

    # (location, period)-indexed departure table: departures[location][period]
    # holds the trips bound for `location` that depart at `period`. This
    # replaces a flat, departure-sorted-then-linearly-scanned trip list per
    # location (O(trips ever bound for location), which grows with horizon)
    # with direct O(1) indexing by period.
    departures::Dict{Node, Vector{Vector{Trip}}}

    # Downstream customers depend only on network topology, which is fixed
    # for the lifetime of an Env, so they're computed once here (reusing a
    # single graph build) instead of on every get_mean_demand call - which
    # otherwise rebuilds the whole location graph from scratch (see
    # create_graph) on every call, a cost that used to repeat on every
    # period of every one of optimize!'s thousands of policy evaluations.
    downstream_customers::Dict{Node, Array{Customer, 1}}

    # Mean demand depends only on downstream_customers and initial_states,
    # both fixed for the lifetime of an Env, so results are memoized here the
    # first time each (location, product, time) is actually asked for -
    # avoiding both the graph rebuild above and the redundant re-summation of
    # the same demand figures across repeated identical queries.
    mean_demand_cache::Dict{Tuple{Node, Product, Int64}, Float64}

    # Whether simulate() should archive the per-period history
    # (historical_on_hand/historical_orders/historical_filled_orders/
    # historical_transportation) that Reporting.jl's get_total_* functions
    # scan after a run. Cost/quantity totals are always available via
    # state.metrics regardless of this flag (see SimMetrics) - this only
    # controls the extra per-period bookkeeping that exists purely to
    # support that after-the-fact scanning (and visualization, which reads
    # the same arrays), so it can be turned off wherever nothing needs it,
    # e.g. optimize!'s thousands of throwaway policy evaluations.
    record_history::Bool

    # Whether simulate() should maintain State's outbound_order_quantities
    # index (see get_past_outbound_orders). Unlike record_history above,
    # this isn't a caller-facing toggle: it's derived from the policies
    # actually in play, via required_lookback (see Policy.jl). It stays
    # false - no per-order bookkeeping at all - unless some policy here
    # (currently only BackwardCoverageOrderingPolicy) actually declares it
    # needs to look backward, so every other policy pays nothing for a
    # capability it never uses.
    needs_outbound_order_index::Bool

    function Env(supplychain::SupplyChain, initial_states, policies; record_history::Bool=true)
        trips = get_trips(supplychain, policies)
        locations = get_locations(supplychain)

        (graph, mapping) = create_graph(supplychain)
        reverse_mapping = Vector{eltype(mapping.keys)}(undef, length(mapping))
        for (k, v) in mapping
            reverse_mapping[v] = k
        end
        sorted_locations = reverse_mapping[topological_sort_by_dfs(graph)]

        downstream_customers = Dict{Node, Array{Customer, 1}}()
        for location in locations
            parents = dfs_parents(graph, mapping[location])
            downstream_customers[location] = Customer[reverse_mapping[i] for i in 1:length(mapping) if parents[i] > 0 && reverse_mapping[i] isa Customer]
        end

        # Group trips by (destination, period) in a single pass instead of,
        # for each location, filtering the entire trip list with
        # is_destination: that pattern is O(locations * lanes * horizon)
        # since |trips| already scales with lanes * horizon, and every trip
        # is revisited once per location. Each trip only needs to be pushed
        # into the (typically 1-2) destinations it actually has, at the
        # period slot it actually departs.
        departures = Dict{Node, Vector{Vector{Trip}}}(location => [Trip[] for _ in 1:supplychain.horizon] for location in locations)
        for trip in trips
            for destination in trip.route.destinations
                if haskey(departures, destination)
                    push!(departures[destination][trip.departure], trip)
                end
            end
        end

        return new(supplychain,
                   collect(initial_states),
                   sorted_locations,
                   collect(supplychain.products),
                   departures,
                   downstream_customers,
                   Dict{Tuple{Node, Product, Int64}, Float64}(),
                   record_history,
                   any(p -> required_lookback(p) > 0, values(policies)))
    end
end

function get_inbound_trips(supplychain, location)
    return collect(sort(filter(trip -> is_destination(location, trip.route), get_trips(supplychain.lanes, supplychain.horizon)), by=t -> t.unit_cost))
end

function get_mean_demands(env::Env)

end

function get_mean_demand(env::Env, customer::Customer, product::Product, time::Int)
    return get!(env.mean_demand_cache, (customer, product, time)) do
        sum(initial_state.demand[(customer, product)].demand[time] for initial_state in env.initial_states) / length(env.initial_states)
    end
end

function get_mean_demand(env::Env, location::Node, product::Product, time::Int)
    return get!(env.mean_demand_cache, (location, product, time)) do
        demand = 0.0
        for customer in env.downstream_customers[location]
            demand = demand + get_mean_demand(env, customer, product, time)
        end
        return demand
    end
end