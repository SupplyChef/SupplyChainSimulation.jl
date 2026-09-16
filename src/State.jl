import Base.push!
import Base.delete!

"""
    _node_country(node)::Union{Nothing, String}

A `ConcreteNode`'s `Location.country` (see `SupplyChainModeling.Location`), or
`nothing` if the node has no `location` set (`Union{Location, Missing}`) or its
`location` has no `country` set.
"""
@inline _node_country(node)::Union{Nothing, String} = ismissing(node.location) ? nothing : node.location.country

"""
Tariff bookkeeping shared by `State` and `Env`. Computed independently by each -
`State` is constructed before `Env` (see `simulate(supplychain, policies)`), so
there's nothing to cache one on the other - but it's cheap (O(products +
suppliers)) and only non-trivial at all when the supply chain actually has
tariffs (`needs_tracking` below): every consumer of `is_relevant` elsewhere
checks it before doing any tariff-specific work, so an untariffed supply chain
pays for one `Bool` per product and a 1-element `origin_countries` and nothing
else.

`origin_countries[1]` is always `nothing` - the "unknown provenance" bucket for
initial inventory/arrivals and any Supplier without a country set (see
`_build_tariff_context`) - mirroring the same convention in
`SupplyChainOptimization.jl`'s re-export tariff handling. `get_tariff_rate`
never charges a tariff against it (see its `isnothing` checks), so
unknown-provenance inventory is simply never taxed on re-export.

`declared_value_by_origin[product_index, origin_country_index]` is the average
`unit_cost` for that product across every `Supplier` in that origin country -
used only for a re-export (Storage-origin) shipment, where cohort tracking
knows which country a unit came from but not which specific source (see
`_remove_on_hand_origin!`). A Supplier-origin shipment instead uses that exact
supplier's own `unit_cost` directly (see `_static_tariff_cost`, Simulation.jl) -
there's no ambiguity about the source in that case. Plants are never a
shipping origin in this simulator (see `get_locations`), so they never
contribute here.
"""
struct TariffContext
    needs_tracking::Bool
    is_relevant::Vector{Bool}                                  # by product_index
    origin_countries::Vector{Union{Nothing, String}}           # origin_countries[1] == nothing
    origin_country_index::Dict{Union{Nothing, String}, Int64}
    declared_value_by_origin::Matrix{Float64}                  # [product_index, origin_country_index]
end

function _build_tariff_context(supply_chain::SupplyChain, product_index::Dict{Product, Int64}, nproducts::Int64)::TariffContext
    if isempty(supply_chain.tariffs)
        empty_index = Dict{Union{Nothing, String}, Int64}(nothing => 1)
        return TariffContext(false, fill(false, nproducts), Union{Nothing, String}[nothing], empty_index, zeros(Float64, nproducts, 1))
    end

    is_relevant = fill(false, nproducts)
    if any(isnothing(t.product) for t in supply_chain.tariffs)
        fill!(is_relevant, true)
    else
        for t in supply_chain.tariffs
            if !isnothing(t.product)
                is_relevant[product_index[t.product]] = true
            end
        end
    end

    origin_countries = Union{Nothing, String}[nothing]
    for supplier in supply_chain.suppliers
        country = _node_country(supplier)
        isnothing(country) || push!(origin_countries, country)
    end
    unique!(origin_countries)
    origin_country_index = Dict{Union{Nothing, String}, Int64}(c => i for (i, c) in enumerate(origin_countries))

    sums = zeros(Float64, nproducts, length(origin_countries))
    counts = zeros(Int64, nproducts, length(origin_countries))
    for supplier in supply_chain.suppliers
        country = _node_country(supplier)
        isnothing(country) && continue
        oci = origin_country_index[country]
        for (product, unit_cost) in supplier.unit_cost
            pi = get(product_index, product, 0)
            (pi == 0 || !is_relevant[pi]) && continue
            sums[pi, oci] += unit_cost
            counts[pi, oci] += 1
        end
    end
    declared_value_by_origin = zeros(Float64, nproducts, length(origin_countries))
    for pi in 1:nproducts, oci in 1:length(origin_countries)
        if counts[pi, oci] > 0
            declared_value_by_origin[pi, oci] = sums[pi, oci] / counts[pi, oci]
        end
    end

    return TariffContext(true, is_relevant, origin_countries, origin_country_index, declared_value_by_origin)
end

"""
Contains information about the historical and current state of the simulation, including inventory positions and pending orders.
"""
mutable struct State
    supply_chain::SupplyChain

    # [location_index, product_index] -> that (location, product) pair's
    # Demand, or `nothing` if it has none (most location/product pairs -
    # only Customers ever have demand). Was a Dict{Tuple{Customer,Product},
    # Demand}: CPU profiling of record_fill! (see its comment) found hashing
    # that compound tuple key as ~85% of record_fill!'s self-time, called
    # once per filled order line - the same Dict-lookup-on-the-hot-path
    # pattern already eliminated from on_hand_inventory/in_transit_inventory/
    # etc. via this same flat-indexing trick, just never applied here.
    demand::Matrix{Union{Nothing, Demand}}

    # Dense integer indices for every storage/product in supply_chain,
    # fixed for the lifetime of a State. Lets the on_hand_* fields below be
    # flat Matrix/Vector-of-Vector containers instead of Dicts keyed by
    # (Storage, Product) tuples - turning every on-hand access into direct
    # array indexing (no hashing, no lazy-allocate-on-miss branch) instead
    # of a Dict lookup. storages/products are the reverse mapping (index ->
    # object), used when a mutation site needs to walk every (storage,
    # product) pair (see snapshot_state!).
    storage_index::Dict{Storage, Int64}
    product_index::Dict{Product, Int64}
    storages::Vector{Storage}
    products::Vector{Product}

    # Same trick, but covering every location in the network (storages,
    # customers, suppliers - see get_locations) rather than just storages:
    # lets in_transit_inventory/pending_*_order_lines/
    # outbound_order_quantities below be flat Matrixes too.
    location_index::Dict{ConcreteNode, Int64}
    locations::Vector{ConcreteNode}

    # Same trick again, covering every lane in the network - lets
    # env.past_orders_buffers (Env.jl) and metrics.seen_trips (Metrics.jl)
    # be indexed directly by lane rather than through a Dict keyed by the
    # Lane object itself.
    lane_index::Dict{Lane, Int64}
    lanes::Vector{Lane}

    # [storage_index, product_index] -> horizon-length array of quantity by
    # age (simulation period the inventory arrived), indexed directly by
    # age instead of through a per-(storage,product) Dict{Int64,Int64} -
    # same flat-array trick already used for in_transit_inventory below.
    on_hand_inventory::Matrix{Vector{Int64}}

    # Ages (simulation periods) that currently have a nonzero-ever
    # on_hand_inventory bucket for a given (storage, product), kept in
    # ascending order. Inventory is only ever added/set at the current,
    # monotonically increasing simulation time (see
    # add_on_hand_inventory!/set_on_hand_inventory!), so a newly-seen age is
    # always >= every age already recorded here - it can just be appended,
    # no sort needed. remove_on_hand_inventory! reads this directly for its
    # FIFO (oldest-first) consumption order instead of collect()-ing and
    # sort!()-ing on_hand_inventory's keys on every call.
    on_hand_ages_order::Matrix{Vector{Int64}}

    # Running total per (storage, product) of everything in the
    # on_hand_inventory age buckets above, kept in sync by every mutation
    # site (add/remove/set/expire/reset). get_on_hand_inventory is called
    # per pending order line inside send_inventory! and by every
    # inventory-position-based policy, and previously re-summed the age
    # buckets (allocating a default Dict on misses) on each call.
    on_hand_totals::Matrix{Int64}

    # [location_index, product_index] -> horizon-length array of quantity in
    # transit, by arrival period - indexed directly instead of through a
    # per-(location, product) Dict, same flat-array trick as on_hand_inventory
    # above.
    in_transit_inventory::Matrix{Vector{Int64}}

    # [storage_index, product_index] -> horizon-length array of overflow
    # quantity by period.
    overflow_inventory::Matrix{Vector{Int64}}

    # Tariff bookkeeping - see TariffContext. Independent of Env (see its
    # docstring for why), but exists on both.
    tariff_context::TariffContext

    # [storage_index, product_index] -> for a tariff-relevant product
    # (tariff_context.is_relevant[pi]), a horizon-length Vector paralleling
    # on_hand_inventory's own age buckets, each entry a Vector{Int64} indexed
    # by tariff_context.origin_country_index giving that (storage, product,
    # age)'s on-hand quantity broken down by origin country - the provenance
    # overlay a Storage-origin shipment needs to price its re-export tariff
    # (see _remove_on_hand_origin!/_cohort_tariff_cost, Simulation.jl). Empty
    # (Vector{Vector{Int64}}()) for every product that isn't tariff-relevant,
    # so on-hand bookkeeping for an untariffed product/network never touches
    # this at all - on_hand_inventory/on_hand_totals above, which every other
    # consumer reads, are completely unaffected by any of this.
    on_hand_by_origin::Matrix{Vector{Vector{Int64}}}

    # Same idea as on_hand_by_origin, but for in_transit_inventory: tags a
    # shipment's provenance from the moment it's sent (send_inventory!) until
    # it's received into on-hand (receive_inventory!, which folds it into
    # on_hand_by_origin above) or delivered to a Customer (which needs no
    # further tracking - see send_inventory!(..., ::Customer, ...)).
    in_transit_by_origin::Matrix{Vector{Vector{Int64}}}

    # [location_index, product_index] -> pending order lines. Vector rather
    # than Set: avoids both Set{OrderLine} hashing overhead and the
    # collect() copy send_inventory! used to pay every period to iterate
    # in creation/due-date order - see send_inventory!'s in-place sort!/filter!.
    pending_outbound_order_lines::Matrix{Vector{OrderLine}}
    pending_inbound_order_lines::Matrix{Vector{OrderLine}}

    # Vector rather than Set: OrderLine has no custom hash/== so a Set falls
    # back to Julia's default identity-based (objectid) hashing, which
    # profiling showed as the single largest self-time cost in optimize!'s
    # hot loop. Order lines are only ever pushed once each (no dedup need),
    # and the one consumer that needs actual set semantics
    # (get_total_lost_sales, Reporting.jl) already wraps the flattened
    # historical data in its own Set(...) regardless of source container
    # type - same reasoning already applied to pending_outbound/inbound_
    # order_lines above.
    filled_orders::Vector{OrderLine}
    placed_orders::Vector{OrderLine}

    historical_on_hand::Array{Dict{Tuple{Storage, Product}, Int64}, 1}
    historical_orders::Array{Vector{OrderLine}, 1}
    historical_transportation::Set{Trip}
    historical_filled_orders::Array{Vector{OrderLine}, 1}
    #historical_pending_outbound_order_lines::Array{Dict{ConcreteNode, Set{OrderLine}}}

    # Incrementally-updated running totals, kept in sync with the
    # historical_* arrays above regardless of Env.record_history - see
    # SimMetrics.
    metrics::SimMetrics

    # [location_index, product_index] -> horizon-length array of quantity
    # ordered per creation_time. Populated only when
    # Env.needs_outbound_order_index is set (i.e. some policy actually
    # declares required_lookback(policy) > 0 - see Policy.jl), but allocated
    # regardless (like every other Matrix field here) since the whole point
    # is to avoid a lazy-allocate-on-miss branch on every write; reading an
    # unpopulated cell just sees the zeros it was initialized with, same as
    # the old Dict's "never touched" miss. Lets get_past_outbound_orders
    # answer "how much did this location ship out at period t" with a
    # direct array read instead of rescanning historical_orders' per-period
    # Sets (which hold every order placed by every location, not just the
    # one being asked about) for a match.
    outbound_order_quantities::Matrix{Vector{Int64}}

    # snapshot_state! sizehint!s each period's fresh on_hand_snapshot Dict to
    # this - the *previous* period's actual final size - instead of either
    # the full dense worst case (every storage x product pair, however few
    # are ever actually touched - the bug removed in favor of an unhinted
    # Dict) or no hint at all (which still pays for several rehash/regrow
    # steps as the Dict grows from empty to its natural size every single
    # period). Consecutive periods in a given simulation tend to touch a
    # similar number of (storage, product) pairs, so last period's size is
    # usually a good estimate of this period's - self-tuning, so it adapts
    # if that changes instead of assuming a fixed worst case. Deliberately
    # not reset by reset!: carrying the estimate across repeated
    # reset!+simulate() cycles on the same reused State (as optimize! does,
    # thousands of times per search) only helps, since the occupancy
    # pattern is a stable property of the network, not of any one trial.
    last_on_hand_snapshot_size::Int64

    function State(supply_chain; pending_outbound_order_lines=Dict{Storage, Array{OrderLine, 1}}())
        # get_storage_index/get_product_index/get_location_index/
        # get_lane_index (SupplyChainModeling.jl) cache the Vector+Dict pair
        # on supply_chain itself, computed once and reused by every
        # State/Env built from it - instead of every State/Env
        # independently re-enumerating the same (read-only, for the
        # duration of a simulation) Sets/Array into an identical Dict, as
        # every scenario's State in optimize! used to do.
        storages_indexed = get_storage_index(supply_chain)
        products_indexed = get_product_index(supply_chain)
        storages = storages_indexed.items
        products = products_indexed.items
        storage_index = storages_indexed.index
        product_index = products_indexed.index
        nstorages = length(storages)
        nproducts = length(products)
        horizon = supply_chain.horizon

        locations_indexed = get_location_index(supply_chain)
        locations = locations_indexed.items
        location_index = locations_indexed.index
        nlocations = length(locations)

        demand = Union{Nothing, Demand}[nothing for _ in 1:nlocations, _ in 1:nproducts]
        for d in supply_chain.demand
            demand[location_index[d.customer], product_index[d.product]] = d
        end

        lanes_indexed = get_lane_index(supply_chain)
        lanes = lanes_indexed.items
        lane_index = lanes_indexed.index

        tariff_context = _build_tariff_context(supply_chain, product_index, nproducts)
        n_origin_countries = length(tariff_context.origin_countries)
        # Full horizon-length age-bucket structure only for a tariff-relevant
        # product; an empty outer Vector otherwise - see on_hand_by_origin's/
        # in_transit_by_origin's field docs above.
        on_hand_by_origin = Matrix{Vector{Vector{Int64}}}(undef, nstorages, nproducts)
        for pi in 1:nproducts, si in 1:nstorages
            on_hand_by_origin[si, pi] = tariff_context.is_relevant[pi] ? [zeros(Int64, n_origin_countries) for _ in 1:horizon] : Vector{Int64}[]
        end
        in_transit_by_origin = Matrix{Vector{Vector{Int64}}}(undef, nlocations, nproducts)
        for pi in 1:nproducts, li in 1:nlocations
            in_transit_by_origin[li, pi] = tariff_context.is_relevant[pi] ? [zeros(Int64, n_origin_countries) for _ in 1:horizon] : Vector{Int64}[]
        end

        state = new(supply_chain,
                   demand,
                   storage_index,
                   product_index,
                   storages,
                   products,
                   location_index,
                   locations,
                   lane_index,
                   lanes,
                   [zeros(Int64, horizon) for _ in 1:nstorages, _ in 1:nproducts],
                   [Int64[] for _ in 1:nstorages, _ in 1:nproducts],
                   zeros(Int64, nstorages, nproducts),
                   [zeros(Int64, horizon) for _ in 1:nlocations, _ in 1:nproducts],
                   [zeros(Int64, horizon) for _ in 1:nstorages, _ in 1:nproducts],
                   tariff_context,
                   on_hand_by_origin,
                   in_transit_by_origin,
                   [OrderLine[] for _ in 1:nlocations, _ in 1:nproducts],
                   [OrderLine[] for _ in 1:nlocations, _ in 1:nproducts],
                   OrderLine[],
                   OrderLine[],
                   [],
                   OrderLine[],
                   Set{Trip}(),
                   [],
                   SimMetrics(length(lanes), horizon),
                   [zeros(Int64, horizon) for _ in 1:nlocations, _ in 1:nproducts],
                   0)
                   #,[])

        reset!(state)

        for order_line in collect(Base.Iterators.flatten(values(pending_outbound_order_lines)))
            add_order_line!(state, order_line)
        end

        return state
    end
end

"""
    reset!(state::State)

Resets `state`'s mutable containers (on-hand/in-transit/overflow inventory, pending
order lines, filled/placed orders, and history) back to the pristine condition of a
freshly constructed `State` for the same supply chain: only the initial on-hand
inventory and in-transit arrivals configured on the supply chain, with no pending
orders and no history.

`state.supply_chain` and `state.demand` are read-only for the duration of a
simulation and are left untouched. This lets the same `State` be re-simulated many
times - as `optimize!` does across thousands of policy evaluations - by resetting
its mutable containers in place instead of `deepcopy`-ing the (potentially large,
read-only) supply chain network on every evaluation.

Any order lines passed via the `pending_outbound_order_lines` keyword at
construction time are a one-time seed and are not restored by `reset!`.
"""
function reset!(state::State)
    for i in eachindex(state.on_hand_inventory)
        fill!(state.on_hand_inventory[i], 0)
        empty!(state.on_hand_ages_order[i])
    end
    fill!(state.on_hand_totals, 0)
    for i in eachindex(state.in_transit_inventory)
        fill!(state.in_transit_inventory[i], 0)
    end
    for i in eachindex(state.overflow_inventory)
        fill!(state.overflow_inventory[i], 0)
    end
    for i in eachindex(state.on_hand_by_origin)
        for v in state.on_hand_by_origin[i]
            fill!(v, 0)
        end
    end
    for i in eachindex(state.in_transit_by_origin)
        for v in state.in_transit_by_origin[i]
            fill!(v, 0)
        end
    end
    for i in eachindex(state.pending_outbound_order_lines)
        empty!(state.pending_outbound_order_lines[i])
        empty!(state.pending_inbound_order_lines[i])
    end
    empty!(state.filled_orders)
    empty!(state.placed_orders)
    empty!(state.historical_on_hand)
    empty!(state.historical_orders)
    empty!(state.historical_transportation)
    empty!(state.historical_filled_orders)
    reset!(state.metrics)
    for i in eachindex(state.outbound_order_quantities)
        fill!(state.outbound_order_quantities[i], 0)
    end

    for storage in state.supply_chain.storages
        for product in state.supply_chain.products
            initial_inventory = get_initial_inventory(storage, product)
            if initial_inventory > 0
                set_on_hand_inventory!(state, storage, product, initial_inventory, 1)
            end
        end
    end

    for lane in state.supply_chain.lanes
        if !isnothing(lane.initial_arrivals)
            for (product, arrivals) in lane.initial_arrivals
                pi = state.product_index[product]
                track_origin = state.tariff_context.is_relevant[pi]
                unknown_index = track_origin ? state.tariff_context.origin_country_index[nothing] : 0
                for i in 1:length(lane.destinations)
                    li = state.location_index[lane.destinations[i]]
                    for j in 1:length(arrivals[i])
                        add_in_transit_inventory!(state, lane.destinations[i], product, j, arrivals[i][j])
                        # Pre-seeded arrivals predate the simulation, same as
                        # initial_inventory above - unknown provenance (see
                        # TariffContext's docstring).
                        if track_origin && arrivals[i][j] != 0
                            state.in_transit_by_origin[li, pi][j][unknown_index] += arrivals[i][j]
                        end
                    end
                end
            end
        end
    end

    return state
end

"""
    get_metrics(state::State)::SimMetrics

Gets the running cost/quantity totals accumulated so far for `state`. See
`SimMetrics`.
"""
function get_metrics(state::State)::SimMetrics
    return state.metrics
end

function add_order_line!(state::State, order_line::OrderLine)
    pi = state.product_index[order_line.product]
    Base.push!(state.pending_outbound_order_lines[state.location_index[order_line.origin], pi], order_line)
    Base.push!(state.pending_inbound_order_lines[state.location_index[order_line.destination], pi], order_line)
end

function delete_order_line!(state::State, order_line::OrderLine)
    pi = state.product_index[order_line.product]

    # Fast deletion from outbound vector
    outbound = state.pending_outbound_order_lines[state.location_index[order_line.origin], pi]
    idx1 = findfirst(==(order_line), outbound)
    if !isnothing(idx1)
        deleteat!(outbound, idx1)
    end

    # Fast deletion from inbound vector
    inbound = state.pending_inbound_order_lines[state.location_index[order_line.destination], pi]
    idx2 = findfirst(==(order_line), inbound)
    if !isnothing(idx2)
        deleteat!(inbound, idx2)
    end
end

# Shared with send_inventory! (Simulation.jl), which processes every pending
# order line for one fixed (location, product) pair per call and so can
# resolve product_index[product] once up front and pass it here, instead of
# every deleted line re-resolving the same product index. destination (and
# so location_index[destination]) still varies per order line - each line's
# customer/downstream location - so that lookup stays per-call.
@inline function _delete_inbound_order_line_by_index!(state::State, order_line::OrderLine, pi::Int64)
    inbound = state.pending_inbound_order_lines[state.location_index[order_line.destination], pi]
    idx = findfirst(==(order_line), inbound)
    if !isnothing(idx)
        deleteat!(inbound, idx)
    end
end

function delete_inbound_order_line!(state::State, order_line::OrderLine)
    _delete_inbound_order_line_by_index!(state, order_line, state.product_index[order_line.product])
end

function delete_order_lines!(state::State, order_lines)
    for order_line in order_lines
        delete_order_line!(state, order_line)
    end
end

# Only ever called (via reset!) once per (storage, product) at time == 1, so
# the ages_order dedup here doesn't need to be O(1) - see add_on_hand_inventory!
# for the hot-path version of the same bookkeeping.
function set_on_hand_inventory!(state::State, to::ConcreteNode, product::Product, quantity, time)
    si = state.storage_index[to]
    pi = state.product_index[product]
    ages = state.on_hand_inventory[si, pi]
    ages_order = state.on_hand_ages_order[si, pi]
    previous = ages[time]
    if time ∉ ages_order
        push!(ages_order, time)
    end
    ages[time] = Int(quantity)
    state.on_hand_totals[si, pi] += Int(quantity) - previous

    if state.tariff_context.is_relevant[pi]
        # Initial inventory predates the simulation - unknown provenance (see
        # TariffContext's docstring). Only ever called once per (storage,
        # product) at time == 1 (see this function's docstring), so `previous`
        # is always 0 in practice - the delta form just mirrors on_hand_totals'
        # own bookkeeping above.
        unknown_index = state.tariff_context.origin_country_index[nothing]
        state.on_hand_by_origin[si, pi][time][unknown_index] += Int(quantity) - previous
    end
end

# Shared with receive_inventory! (Simulation.jl), which resolves si/li/pi
# once for a given (location, product) and calls this + the other _by_index
# writers below directly, instead of add_on_hand_inventory!/
# add_in_transit_inventory!/get_on_hand_inventory/get_in_transit_inventory/
# record_overflow! each independently re-resolving the same indices.
#
# receive_inventory! calls this once per (storage, product) every period
# regardless of whether anything is actually arriving (quantity == 0 is the
# common case for most (storage, product, period) combinations on a large,
# sparsely-active network) - recording an age for a zero-quantity "arrival"
# would just be dead weight every downstream ages_order reader
# (_remove_on_hand_by_index!, expire_on_hand_inventory, snapshot_state!'s
# "was this pair ever touched" check) has to scan past for no benefit, so
# skip it entirely when there's nothing to record.
@inline function _add_on_hand_by_index!(state::State, si::Int64, pi::Int64, quantity::Int64, time)
    if quantity > 0
        ages = state.on_hand_inventory[si, pi]
        ages_order = state.on_hand_ages_order[si, pi]
        # Ages are only ever touched at the current, monotonically
        # increasing simulation time, so a new age is only ever the *last*
        # one appended (or the very first) - checking ages_order's tail is
        # O(1) and equivalent to the old per-call Dict haskey check.
        if isempty(ages_order) || ages_order[end] != time
            push!(ages_order, time)
        end
        ages[time] += quantity
        state.on_hand_totals[si, pi] += quantity
    end
end

function add_on_hand_inventory!(state::State, to::Storage, product::Product, quantity::Int64, time)
    si = state.storage_index[to]
    pi = state.product_index[product]
    _add_on_hand_by_index!(state, si, pi, quantity, time)
end

# Shared with send_inventory!'s ConcreteNode method (Simulation.jl), which
# resolves si/pi once for a (location, product) it may fulfil several order
# lines from in a single call, and calls this once per fulfilled line instead
# of each call re-resolving storage_index[to]/product_index[product] via its
# own Dict lookup.
@inline function _remove_on_hand_by_index!(state::State, si::Int64, pi::Int64, quantity::Int64)
    ages = state.on_hand_inventory[si, pi]
    removed_total = 0
    # FIFO: must consume oldest inventory (smallest age/arrival time) first.
    # on_hand_ages_order already holds every age ever seen for this
    # (storage, product) in ascending order (ages only ever get added at the
    # current, monotonically increasing simulation time), so it can be
    # iterated directly instead of collect()-ing and sort!()-ing
    # on_hand_inventory's keys on every call.
    for t in state.on_hand_ages_order[si, pi]
        if quantity <= 0
            break
        end
        removed_quantity = min(quantity, ages[t])
        ages[t] -= removed_quantity
        quantity -= removed_quantity
        removed_total += removed_quantity
    end
    state.on_hand_totals[si, pi] -= removed_total
end

function remove_on_hand_inventory!(state::State, to::Storage, product::Product, quantity::Int64)
    si = state.storage_index[to]
    pi = state.product_index[product]
    _remove_on_hand_by_index!(state, si, pi, quantity)
end

# Origin-country provenance overlay for tariff-relevant products (see
# TariffContext). Every function below is only ever called from Simulation.jl
# at a site that already checked state.tariff_context.is_relevant[pi] first -
# an untariffed (storage, product) pair never reaches any of this.

@inline function _add_on_hand_origin!(state::State, si::Int64, pi::Int64, time::Int64, by_origin::Vector{Int64})
    bucket = state.on_hand_by_origin[si, pi][time]
    @inbounds for i in eachindex(by_origin)
        bucket[i] += by_origin[i]
    end
end

"""
    _remove_on_hand_origin!(state, si, pi, quantity)::Vector{Int64}

Removes `quantity` units' worth of origin-country provenance from
`on_hand_by_origin[si, pi]`, in lockstep with `_remove_on_hand_by_index!`'s own
FIFO (oldest-age-first) walk - within an age, countries are consumed in a
fixed order (`tariff_context.origin_country_index`'s ordering). Returns the
breakdown of what was removed (sums to `quantity`, under the same
`on_hand_totals[si, pi] >= quantity` precondition `_remove_on_hand_by_index!`
already relies on) - this is what a Storage-origin shipment tags its in-transit
addition with (see `send_inventory!`/`_cohort_tariff_cost`, Simulation.jl).

Called separately from, but consuming the same ages as, the plain
`_remove_on_hand_by_index!` - independent because they walk two separate
arrays (on_hand_by_origin vs. on_hand_inventory/on_hand_totals), each fully
self-consistent as long as nothing else mutates either between the two calls
(true here: this code is single-threaded and sequential).
"""
function _remove_on_hand_origin!(state::State, si::Int64, pi::Int64, quantity::Int64)::Vector{Int64}
    by_origin = state.on_hand_by_origin[si, pi]
    removed = zeros(Int64, length(state.tariff_context.origin_countries))
    remaining = quantity
    for t in state.on_hand_ages_order[si, pi]
        remaining <= 0 && break
        bucket = by_origin[t]
        @inbounds for oci in eachindex(bucket)
            remaining <= 0 && break
            take = min(remaining, bucket[oci])
            take == 0 && continue
            bucket[oci] -= take
            removed[oci] += take
            remaining -= take
        end
    end
    return removed
end

@inline function _add_in_transit_origin!(state::State, li::Int64, pi::Int64, time::Int64, by_origin::Vector{Int64})
    bucket = state.in_transit_by_origin[li, pi][time]
    @inbounds for i in eachindex(by_origin)
        bucket[i] += by_origin[i]
    end
end

"""
    _remove_in_transit_origin!(state, li, pi, time, quantity)::Vector{Int64}

Removes `quantity` units' worth of origin-country provenance from
`in_transit_by_origin[li, pi][time]`, consuming countries in a fixed
(`origin_country_index`) order - there's no age/batch ordering to preserve
here the way `_remove_on_hand_origin!` has (a single (location, product, time)
in-transit bucket isn't itself split by arrival batch, see
`in_transit_inventory`). Used by `receive_inventory!` to split a partially-
accepted arrival (the rest overflows to next period, see
`_record_overflow_by_index!`) into "accepted" and "still in transit"
breakdowns. Returns the breakdown of what was removed.
"""
function _remove_in_transit_origin!(state::State, li::Int64, pi::Int64, time::Int64, quantity::Int64)::Vector{Int64}
    bucket = state.in_transit_by_origin[li, pi][time]
    removed = zeros(Int64, length(bucket))
    remaining = quantity
    @inbounds for oci in eachindex(bucket)
        remaining <= 0 && break
        take = min(remaining, bucket[oci])
        take == 0 && continue
        bucket[oci] -= take
        removed[oci] += take
        remaining -= take
    end
    return removed
end

"""
    _static_origin_breakdown(state, country, quantity)::Vector{Int64}

A one-hot origin-country breakdown: all `quantity` units attributed to
`country` (or the unknown-provenance bucket if `country` is `nothing`). Used
for a Supplier-origin shipment, where every unit ships from the same static
country - no cohort tracking needed (see `TariffContext`'s docstring).
"""
function _static_origin_breakdown(state::State, country::Union{Nothing, String}, quantity::Int64)::Vector{Int64}
    v = zeros(Int64, length(state.tariff_context.origin_countries))
    oci = get(state.tariff_context.origin_country_index, country, state.tariff_context.origin_country_index[nothing])
    v[oci] = quantity
    return v
end

# Shared by get_on_hand_inventory and get_net_inventory - the latter used to
# call get_on_hand_inventory/get_in_transit_inventories/get_inbound_orders/
# get_outbound_orders, each independently re-resolving storage_index[to]/
# location_index[to]/product_index[product] via a Dict lookup for the same
# (location, product) pair - 8 Dict lookups to answer one get_net_inventory
# query. Splitting the by-index logic out lets get_net_inventory resolve
# each index exactly once and pass it to all four, while these public
# functions (still used standalone elsewhere) keep their own signatures.
@inline function _on_hand_by_index(state::State, si::Int64, pi::Int64)::Int64
    (si == 0 || pi == 0) && return 0
    return state.on_hand_totals[si, pi]
end

function get_on_hand_inventory(state::State, to::ConcreteNode, product::Product)::Int64
    si = get(state.storage_index, to, 0)
    if si == 0
        return 0
    end
    pi = get(state.product_index, product, 0)
    return _on_hand_by_index(state, si, pi)
end

# si/pi are resolved once per (location, product) for the whole
# simulate() call (see simulate() in Simulation.jl) and passed in here,
# instead of this - called once per Storage per product per period of
# every simulate() run - independently re-resolving storage_index[to]/
# product_index[product] via its own Dict lookup every time.
function expire_on_hand_inventory(state::State, to::Storage, product::Product, si::Int64, pi::Int64, time)
    max_age = get_maximum_age(to, product)
    ages = state.on_hand_inventory[si, pi]
    track_origin = state.tariff_context.is_relevant[pi]
    by_origin = track_origin ? state.on_hand_by_origin[si, pi] : nothing
    expired_total = 0
    for t in state.on_hand_ages_order[si, pi]
        if t <= time - max_age
            on_hand = ages[t]
            if on_hand > 0
                ages[t] = 0
                expired_total += on_hand
                # Written off along with the plain quantity above - otherwise
                # a future FIFO removal (_remove_on_hand_origin!) would still
                # see "stock" at an age on_hand_inventory itself already
                # considers empty, double-counting provenance that no longer
                # physically exists.
                track_origin && fill!(by_origin[t], 0)
            end
        else
            break
        end
    end
    if expired_total > 0
        state.on_hand_totals[si, pi] -= expired_total
    end
    return
end

@inline function _add_in_transit_by_index!(state::State, li::Int64, pi::Int64, time::Int64, quantity::Int64)
    state.in_transit_inventory[li, pi][time] += quantity
end

function add_in_transit_inventory!(state::State, to::N, product::Product, time::Int64, quantity::Int64) where N <: ConcreteNode
    li = state.location_index[to]
    pi = state.product_index[product]
    _add_in_transit_by_index!(state, li, pi, time, quantity)
end

function delete_in_transit_inventory!(state::State, to::N, product::Product, time::Int64, quantity::Int64) where N <: ConcreteNode
    li = state.location_index[to]
    pi = state.product_index[product]
    state.in_transit_inventory[li, pi][time] -= quantity
end

@inline function _in_transit_by_index(state::State, li::Int64, pi::Int64, time::Int64)::Int64
    (li == 0 || pi == 0) && return 0
    return state.in_transit_inventory[li, pi][time]
end

"""
    get_in_transit_inventory(state::State, to::Location, product::Product, time::Int64)::Int64

    Gets the number of units of a product in transit to a location at a given time.
"""
function get_in_transit_inventory(state::State, to::N, product::Product, time::Int64)::Int64 where N <: ConcreteNode
    li = get(state.location_index, to, 0)
    pi = li == 0 ? 0 : get(state.product_index, product, 0)
    return _in_transit_by_index(state, li, pi, time)
end

# Shared fallback for get_in_transit_inventories misses (a `to`/`product`
# outside the network entirely - see location_index/product_index).
# Callers only ever read from the result, so handing every miss the same
# array (instead of allocating a fresh `[0]` per call) is safe.
const _no_in_transit = Int64[0]

function get_in_transit_inventories(state::State, to::N, product::Product)::Array{Int64, 1} where N <: ConcreteNode
    li = get(state.location_index, to, 0)
    if li == 0
        return _no_in_transit
    end
    pi = get(state.product_index, product, 0)
    if pi == 0
        return _no_in_transit
    end
    return state.in_transit_inventory[li, pi]
end

@inline function _record_overflow_by_index!(state::State, si::Int64, pi::Int64, to::Storage, product::Product, time::Int64, quantity::Int64)
    overflow = state.overflow_inventory[si, pi]
    # This slot is overwritten, not accumulated (receive_inventory! can call
    # record_overflow! more than once for the same (to, product, time) in a
    # period), so the metrics update must track the delta rather than adding
    # the new quantity outright - otherwise a corrected/overwritten value
    # would be double-counted relative to get_total_overflow_costs, which
    # only ever sees the final value left in the slot.
    previous_quantity = overflow[time]
    overflow[time] = quantity
    state.metrics.overflow_costs += (quantity - previous_quantity) * get_overflow_cost(to, product)
end

"""
    record_overflow!(state::State, to::Storage, product::Product, time::Int64, quantity::Int64)

Records that `quantity` units of `product` could not be received into `to`'s on-hand
inventory at `time` because it would have exceeded `maximum_units`, and are being held
in temporary overflow storage instead (see `get_total_overflow_costs`).
"""
function record_overflow!(state::State, to::Storage, product::Product, time::Int64, quantity::Int64)
    si = state.storage_index[to]
    pi = state.product_index[product]
    _record_overflow_by_index!(state, si, pi, to, product, time, quantity)
end

"""
    get_overflow_inventory(state::State, to::Storage, product::Product, time::Int64)::Int64

Gets the number of units of `product` held in temporary overflow storage at `to` at `time`.
"""
function get_overflow_inventory(state::State, to::Storage, product::Product, time::Int64)::Int64
    si = get(state.storage_index, to, 0)
    if si == 0
        return 0
    end
    pi = get(state.product_index, product, 0)
    if pi == 0
        return 0
    end
    return state.overflow_inventory[si, pi][time]
end

"""
    get_horizon(state::State)

    Gets the number of steps in the simulation.
"""
function get_horizon(state::State)
    #return maximum(length.(map(d -> d.demand, collect(state.supply_chain.demand))))
    return state.supply_chain.horizon
end

"""
    snapshot_state!(state::State, time, record_history::Bool, customer_backlog::Bool)

Closes out period `time`: charges holding cost for everything currently on
hand and backlog cost (raw units, see `SimMetrics.backlog`) for everything
currently outstanding (both always, regardless of `record_history` - this is
the incremental counterpart of `get_total_holding_costs`'s history scan, see
`SimMetrics`), and, only when `record_history` is `true`, archives a
per-period on-hand snapshot plus this period's filled/placed orders into
`state`'s `historical_*` arrays for later reporting/visualization.

When `record_history` is `false` the archiving - including the Dict copy of
on-hand inventory and the per-period Set handoffs - is skipped entirely, and
`state.filled_orders`/`state.placed_orders` are just cleared in place for the
next period instead of being swapped for fresh, permanently-retained Sets.

`customer_backlog` must be passed the same value as the `Env` this state is
being simulated under (see `Env.customer_backlog`): when `false` (the
default), a Customer order still pending here is one period away from being
dropped as a lost sale (see `place_orders(..., ::Customer, ...)`), not a
genuine backlog, and is excluded from `SimMetrics.backlog` accordingly; when
`true`, Customer orders queue like any other node's and are counted the same
way.
"""
function snapshot_state!(state::State, time, record_history::Bool, customer_backlog::Bool)
    # A fresh Dict is needed every period regardless of sizing (it's handed
    # to historical_on_hand below and must outlive this call, so it can't be
    # a buffer that's cleared and reused in place period over period). Not
    # sizehint!'d to length(state.on_hand_totals) (every storage x product
    # pair) - that was the full dense worst case, and CPU profiling of the
    # large-network benchmark found allocating/rehashing a table sized for
    # every pair, even though only a fraction are ever actually touched (see
    # the "was this pair ever touched" skip below), as a real chunk of
    # simulate()'s time. sizehint!'d instead to state.last_on_hand_snapshot_size
    # - last period's actual final size (see State's field comment) - which
    # avoids most of the same rehash/regrow cost without resurrecting the
    # dense-worst-case bug: still just an estimate that Dict's own
    # incremental growth strategy can correct if it's wrong, not a
    # guaranteed final size.
    on_hand_snapshot = if record_history
        d = Dict{Tuple{Storage, Product}, Int64}()
        sizehint!(d, state.last_on_hand_snapshot_size)
        d
    else
        nothing
    end
    # on_hand_totals is maintained incrementally by every on-hand mutation
    # site, so per-(storage,product) totals are read directly instead of
    # re-summing each cell's age buckets here every period.
    for pi in 1:size(state.on_hand_totals, 2), si in 1:size(state.on_hand_totals, 1)
        on_hand = state.on_hand_totals[si, pi]
        # A (storage, product) pair that was never touched by any on-hand
        # mutation (on_hand_ages_order empty) never had a corresponding key
        # in the old Dict-backed on_hand_totals either - skip it here too,
        # so historical_on_hand/Visualization keep seeing only pairs that
        # were ever actually stocked.
        if on_hand != 0 || !isempty(state.on_hand_ages_order[si, pi])
            location = state.storages[si]
            product = state.products[pi]
            if on_hand != 0
                state.metrics.holding_costs += on_hand * get(location.unit_holding_cost, product, 0)
            end
            if record_history
                on_hand_snapshot[(location, product)] = on_hand
            end
        end
    end

    # Backlog: every location's currently-outstanding order-line quantity,
    # read directly from pending_outbound_order_lines rather than scanned
    # from history - an order line is removed from there the instant it's
    # filled or dropped (see send_inventory!/record_drop!), so whatever's
    # still sitting there when a period closes out is exactly what's
    # genuinely still owed, no history needed. Always charged, regardless of
    # record_history, same as holding_costs above - see SimMetrics.backlog.
    # Customer-destined order lines only count when customer_backlog is true
    # (see this function's docstring and Env.customer_backlog) - otherwise a
    # still-pending Customer order here is one period away from becoming a
    # lost sale, not a genuine backlog.
    for pi in 1:size(state.pending_outbound_order_lines, 2), li in 1:size(state.pending_outbound_order_lines, 1)
        for ol in state.pending_outbound_order_lines[li, pi]
            if customer_backlog || !isa(ol.destination, Customer)
                state.metrics.backlog += ol.quantity
            end
        end
    end

    if record_history
        state.last_on_hand_snapshot_size = length(on_hand_snapshot)
        push!(state.historical_on_hand, on_hand_snapshot)

        # state.filled_orders/placed_orders are only ever mutated via push! on the
        # field itself, so handing the current Vector to history and replacing the
        # field with a fresh one is equivalent to copy+empty! without the O(n)
        # element-by-element copy.
        push!(state.historical_filled_orders, state.filled_orders)
        state.filled_orders = OrderLine[]

        push!(state.historical_orders, state.placed_orders)
        state.placed_orders = OrderLine[]
    else
        empty!(state.filled_orders)
        empty!(state.placed_orders)
    end
    #push!(state.historical_pending_outbound_order_lines, Dict(k => copy(v) for (k, v) in state.order_line_tracker.pending_inbound_order_lines))
end

@inline function _in_transit_sum_by_index(state::State, li::Int64, pi::Int64, time::Int64)::Int64
    (li == 0 || pi == 0) && return 0
    vec = state.in_transit_inventory[li, pi]
    total = 0
    @inbounds for t in time:length(vec)
        total += vec[t]
    end
    return total
end

@inline function _inbound_orders_by_index(state::State, li::Int64, pi::Int64, time::Int64)::Int64
    (li == 0 || pi == 0) && return 0
    total = 0
    vec = state.pending_inbound_order_lines[li, pi]
    @inbounds for i in eachindex(vec)
        ol = vec[i]
        if ol.due_date >= time
            total += ol.quantity
        end
    end
    return total
end

@inline function _outbound_orders_by_index(state::State, li::Int64, pi::Int64, time::Int64)::Int64
    (li == 0 || pi == 0) && return 0
    total = 0
    vec = state.pending_outbound_order_lines[li, pi]
    @inbounds for i in eachindex(vec)
        ol = vec[i]
        if ol.due_date >= time
            total += ol.quantity
        end
    end
    return total
end

@inline function _net_inventory_by_index(state::State, li::Int64, pi::Int64, si::Int64, time::Int64)
    on_hand = _on_hand_by_index(state, si, pi)
    in_transit = _in_transit_sum_by_index(state, li, pi, time)
    inbound = _inbound_orders_by_index(state, li, pi, time)
    outbound = _outbound_orders_by_index(state, li, pi, time)

    #@debug "on hand: $on_hand, in transit: $in_transit, inbound: $inbound, outbound: $outbound"

    return on_hand +
            in_transit +
            inbound -
            outbound
end

function get_net_inventory(state::State, location::ConcreteNode, product::Product, time::Int64)
    # on-hand + in-transit + on-order from suppliers - on-order from supplied
    #
    # Resolves location_index/product_index/storage_index exactly once and
    # shares them across all four components of _net_inventory_by_index,
    # instead of calling get_on_hand_inventory/get_in_transit_inventories/
    # get_inbound_orders/get_outbound_orders - each of which independently
    # re-resolved the same (location, product) pair via its own pair of Dict
    # lookups. That was 8 Dict lookups to answer one get_net_inventory
    # query; this is 3 (li, pi, and si only when location is actually a
    # Storage).
    li = get(state.location_index, location, 0)
    pi = get(state.product_index, product, 0)
    si = location isa Storage ? get(state.storage_index, location, 0) : 0
    return _net_inventory_by_index(state, li, pi, si, time)
end

"""
    get_inbound_orders(state::State, location::Location, product::Product, time::Int64)::Int64

    Gets the number of units of a product on order to a location (but not yet shipped there) at a given time.
"""
function get_inbound_orders(state::State, location::ConcreteNode, product::Product, time::Int64)::Int64
    li = get(state.location_index, location, 0)
    if li == 0
        return 0
    end
    pi = get(state.product_index, product, 0)
    return _inbound_orders_by_index(state, li, pi, time)
end

"""
    get_outbound_orders(state::State, location::Location, product::Product, time::Int64)::Int64

    Gets the number of units of a product on order at a location (and not yet shipped out) at a given time.
"""
function get_outbound_orders(state::State, location::ConcreteNode, product::Product, time::Int64)::Int64
    li = get(state.location_index, location, 0)
    if li == 0
        return 0
    end
    pi = get(state.product_index, product, 0)
    return _outbound_orders_by_index(state, li, pi, time)
end

"""
    get_past_outbound_orders(state::State, location::ConcreteNode, product::Product, time::Int64, step_back::Int64)::Array{Union{Missing, Int64}, 1}

Gets, for each of the `step_back` periods before `time`, the quantity of
`product` that was ordered *from* `location` (i.e. `location` was the
`origin`) - `missing` for any period before the simulation started.

Reads `state.outbound_order_quantities`, which is only populated when some
policy declares `required_lookback(policy) > 0` (see `Policy.jl`/`Env`); for
a `location`/`product` nothing was ever recorded for (either because no such
policy is in play, or simply because no order ever originated there), every
period reads back as `0`, matching what an exhaustive scan would have found.
"""
function get_past_outbound_orders(state::State, location::ConcreteNode, product::Product, time::Int64, step_back::Int64)::Array{Union{Missing, Int64}, 1}
    past_orders = Array{Union{Missing, Int64}, 1}(undef, step_back)
    return get_past_outbound_orders!(past_orders, state, location, product, time)
end

"""
    get_past_outbound_orders!(past_orders, state, location, product, time)::Array{Union{Missing, Int64}, 1}

Same as `get_past_outbound_orders`, but fills the caller-provided `past_orders`
buffer in place instead of allocating a fresh one - `step_back` is implicitly
`length(past_orders)`. `BackwardCoverageOrderingPolicy.get_order` (Policy.jl)
calls this with a buffer it owns and reuses across every call (`cover`'s
length, and therefore this buffer's size, never changes after construction),
since a fresh `zeros(...)` allocation on every single `get_order` call -
called ~15000 trials x 30 scenarios x every period in `optimize!`'s search -
showed up as a real, avoidable chunk of allocation profiling.
"""
@inline function _fill_past_outbound_orders_by_index!(past_orders::Array{Union{Missing, Int64}, 1}, state::State, li::Int64, pi::Int64, time::Int64)::Array{Union{Missing, Int64}, 1}
    @inbounds for t in eachindex(past_orders)
        creation_time = time - t
        if creation_time < 0
            past_orders[t] = missing
        elseif creation_time == 0 || pi == 0
            # creation_time == 0 predates period 1 (nothing is ever placed
            # then), same as the pre-simulation snapshot the old
            # historical_orders-scanning implementation read as empty here -
            # not `missing`, unlike creation_time < 0 above. li/pi == 0
            # (location/product outside the network) reads back as 0 too:
            # outbound_order_quantities is always allocated (see its field
            # doc), so within the network every period genuinely read as 0
            # unless record_placement! actually wrote a nonzero quantity.
            past_orders[t] = 0
        else
            past_orders[t] = state.outbound_order_quantities[li, pi][creation_time]
        end
    end
    past_orders
end

function get_past_outbound_orders!(past_orders::Array{Union{Missing, Int64}, 1}, state::State, location::ConcreteNode, product::Product, time::Int64)::Array{Union{Missing, Int64}, 1}
    li = get(state.location_index, location, 0)
    pi = li == 0 ? 0 : get(state.product_index, product, 0)
    return _fill_past_outbound_orders_by_index!(past_orders, state, li, pi, time)
end

