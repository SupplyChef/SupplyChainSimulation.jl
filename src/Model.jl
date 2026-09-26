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

    # Ad-valorem tariff cost charged against this order line at fill time (see
    # record_fill!, Simulation.jl) - 0.0 unless the line's product is tariff-
    # relevant. Defaulted so every existing 7-arg call site is unaffected;
    # exists (rather than reading state.metrics.tariff_costs alone) so
    # get_total_tariff_costs (Reporting.jl) can independently scan
    # historical_filled_orders and cross-check state.metrics.tariff_costs, the
    # same way every other cost bucket is cross-checked - see
    # metrics-equivalence-tests.jl. Unlike trip_unit_cost (derivable purely
    # from trip.route.unit_cost * quantity, static Lane data), a tariff's
    # origin-country breakdown is a run-time-only fact - there's no static
    # field to recompute it from after the fact - so it has to be captured here
    # instead.
    tariff_cost::Float64

    OrderLine(creation_time, origin, destination, product, quantity, due_date, trip, tariff_cost=0.0) =
        new(creation_time, origin, destination, product, quantity, due_date, trip, tariff_cost)
end

"""
    ResponseProfile(wait_share, substitute_share, buy_elsewhere_share, defect_share)

Describes how a specific (customer, product) pair actually behaves when an
order for that pair isn't filled the period it's created, replacing the
network-wide `Env.customer_backlog` flag for just that pair (see
`simulate`'s `response_profiles` keyword):

  - `wait_share`: the order backlogs and waits for stock - the exact same
    mechanism `customer_backlog=true` already uses (`due_date` extended to
    `typemax(Int64)`), just applied per-pair instead of network-wide.
  - `substitute_share` / `buy_elsewhere_share`: the order is lost this
    period, identical to the default (`customer_backlog=false`) behavior.
    This package has no product-substitution mapping anywhere in its data
    model, so "substitute" and "buy elsewhere" are mechanically
    indistinguishable here - both are simply a lost sale.
  - `defect_share`: the order is lost this period, *and* this (customer,
    product) pair's demand is permanently zeroed for the remainder of the
    simulation run (no more orders are ever placed for it again).

The outcome for a given unfilled order line is sampled from these shares
exactly once, at the moment that line is first confirmed unfulfilled (see
`Simulation.jl`). The four shares must be non-negative and sum to 1 (within
`1e-6`).
"""
struct ResponseProfile
    wait_share::Float64
    substitute_share::Float64
    buy_elsewhere_share::Float64
    defect_share::Float64

    function ResponseProfile(wait_share, substitute_share, buy_elsewhere_share, defect_share)
        shares = (wait_share, substitute_share, buy_elsewhere_share, defect_share)
        any(s -> s < 0, shares) && throw(ArgumentError("ResponseProfile shares must be non-negative, got $(shares)"))
        isapprox(sum(shares), 1.0; atol=1e-6) || throw(ArgumentError("ResponseProfile shares must sum to 1, got $(sum(shares))"))
        return new(wait_share, substitute_share, buy_elsewhere_share, defect_share)
    end
end

"""
    _sample_outcome(profile::ResponseProfile)::Symbol

Samples one of `:wait`, `:substitute`, `:buy_elsewhere`, `:defect` from
`profile`'s four shares. Falls through to `:defect` once every prior share
has been consumed, so floating-point rounding of a sum that's only
guaranteed to be `1 ± 1e-6` can never leave a probability gap unresolved.
"""
@inline function _sample_outcome(profile::ResponseProfile)::Symbol
    r = rand()
    r < profile.wait_share && return :wait
    r -= profile.wait_share
    r < profile.substitute_share && return :substitute
    r -= profile.substitute_share
    r < profile.buy_elsewhere_share && return :buy_elsewhere
    return :defect
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

`@inline`d: this returns `Union{Trip, Nothing}`, and `Trip` isn't isbits (it
holds `route::Lane` and `policies::Union{Missing, Dict}`, both reference
fields - see `Trip`'s definition). A Union with a non-isbits member can't
use Julia's compact isbits-union representation, so across a real,
non-inlined call boundary Julia has to box the `Trip` on every successful
match. Inlining keeps the Union-typed value local to the (already
concretely-typed) caller instead, letting the compiler union-split it
without boxing - allocation profiling found this as the largest single
allocation site in a full `beer_game()` run.
"""
@inline function find_next_departure(env, destination::ConcreteNode, time::Int64)::Trip
    periods = env.departures[destination]
    for t in time:length(periods)
        trips_at_t = periods[t]
        if !isempty(trips_at_t)
            return trips_at_t[1]
        end
    end
    return NULL_TRIP
end

"""
    find_next_departure(env, destination, time, due_date)

Finds the earliest trip bound for `destination` that departs at or after
`time` and still arrives by `due_date`, or `nothing` if none do. Only scans
the `[time, due_date]` window of `env.departures[destination]`, instead of
the full, unbounded list of trips ever bound for `destination`.

`@inline`d for the same reason as the 3-argument method above - see its
docstring.
"""
@inline function find_next_departure(env, destination::ConcreteNode, time::Int64, due_date::Int64)::Trip
    periods = env.departures[destination]
    last_period = min(due_date, length(periods))
    for t in time:last_period
        for trip in periods[t]
            if t + trip.route.times[1] <= due_date
                return trip
            end
        end
    end
    return NULL_TRIP
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