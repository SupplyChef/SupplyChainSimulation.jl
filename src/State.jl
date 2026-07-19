import Base.push!
import Base.delete!

"""
Contains information about the historical and current state of the simulation, including inventory positions and pending orders.
"""
mutable struct State
    supply_chain::SupplyChain
    demand::Dict{Tuple{Customer, Product}, Demand}

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

    function State(supply_chain; pending_outbound_order_lines=Dict{Storage, Array{OrderLine, 1}}())
        demand = Dict((d.customer, d.product) => d for d in supply_chain.demand)

        storages = collect(supply_chain.storages)
        products = collect(supply_chain.products)
        storage_index = Dict{Storage, Int64}(s => i for (i, s) in enumerate(storages))
        product_index = Dict{Product, Int64}(p => i for (i, p) in enumerate(products))
        nstorages = length(storages)
        nproducts = length(products)
        horizon = supply_chain.horizon

        locations = collect(ConcreteNode, get_locations(supply_chain))
        location_index = Dict{ConcreteNode, Int64}(l => i for (i, l) in enumerate(locations))
        nlocations = length(locations)

        state = new(supply_chain,
                   demand,
                   storage_index,
                   product_index,
                   storages,
                   products,
                   location_index,
                   locations,
                   [zeros(Int64, horizon) for _ in 1:nstorages, _ in 1:nproducts],
                   [Int64[] for _ in 1:nstorages, _ in 1:nproducts],
                   zeros(Int64, nstorages, nproducts),
                   [zeros(Int64, horizon) for _ in 1:nlocations, _ in 1:nproducts],
                   [zeros(Int64, horizon) for _ in 1:nstorages, _ in 1:nproducts],
                   [OrderLine[] for _ in 1:nlocations, _ in 1:nproducts],
                   [OrderLine[] for _ in 1:nlocations, _ in 1:nproducts],
                   OrderLine[],
                   OrderLine[],
                   [],
                   OrderLine[],
                   Set{Trip}(),
                   [],
                   SimMetrics(),
                   [zeros(Int64, horizon) for _ in 1:nlocations, _ in 1:nproducts])
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
                for i in 1:length(lane.destinations)
                    for j in 1:length(arrivals[i])
                        add_in_transit_inventory!(state, lane.destinations[i], product, j, arrivals[i][j])
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

function delete_inbound_order_line!(state::State, order_line::OrderLine)
    # Fast deletion from inbound vector
    inbound = state.pending_inbound_order_lines[state.location_index[order_line.destination], state.product_index[order_line.product]]
    idx = findfirst(==(order_line), inbound)
    if !isnothing(idx)
        deleteat!(inbound, idx)
    end
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
end

function add_on_hand_inventory!(state::State, to::Storage, product::Product, quantity::Int64, time)
    si = state.storage_index[to]
    pi = state.product_index[product]
    ages = state.on_hand_inventory[si, pi]
    ages_order = state.on_hand_ages_order[si, pi]
    # Ages are only ever touched at the current, monotonically increasing
    # simulation time, so a new age is only ever the *last* one appended
    # (or the very first) - checking ages_order's tail is O(1) and
    # equivalent to the old per-call Dict haskey check.
    if isempty(ages_order) || ages_order[end] != time
        push!(ages_order, time)
    end
    ages[time] += quantity
    state.on_hand_totals[si, pi] += quantity
end

function remove_on_hand_inventory!(state::State, to::Storage, product::Product, quantity::Int64)
    si = state.storage_index[to]
    pi = state.product_index[product]
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

function get_on_hand_inventory(state::State, to::ConcreteNode, product::Product)::Int64
    si = get(state.storage_index, to, 0)
    if si == 0
        return 0
    end
    pi = get(state.product_index, product, 0)
    if pi == 0
        return 0
    end
    return state.on_hand_totals[si, pi]
end

function expire_on_hand_inventory(state::State, to::Storage, product::Product, time)
    max_age = get_maximum_age(to, product)
    si = state.storage_index[to]
    pi = state.product_index[product]
    ages = state.on_hand_inventory[si, pi]
    expired_total = 0
    for t in state.on_hand_ages_order[si, pi]
        if t <= time - max_age
            on_hand = ages[t]
            if on_hand > 0
                ages[t] = 0
                expired_total += on_hand
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

function add_in_transit_inventory!(state::State, to::N, product::Product, time::Int64, quantity::Int64) where N <: ConcreteNode
    li = state.location_index[to]
    pi = state.product_index[product]
    state.in_transit_inventory[li, pi][time] += quantity
end

function delete_in_transit_inventory!(state::State, to::N, product::Product, time::Int64, quantity::Int64) where N <: ConcreteNode
    li = state.location_index[to]
    pi = state.product_index[product]
    state.in_transit_inventory[li, pi][time] -= quantity
end

"""
    get_in_transit_inventory(state::State, to::Location, product::Product, time::Int64)::Int64

    Gets the number of units of a product in transit to a location at a given time.
"""
function get_in_transit_inventory(state::State, to::N, product::Product, time::Int64)::Int64 where N <: ConcreteNode
    li = get(state.location_index, to, 0)
    if li == 0
        return 0
    end
    pi = get(state.product_index, product, 0)
    if pi == 0
        return 0
    end
    return state.in_transit_inventory[li, pi][time]
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

"""
    record_overflow!(state::State, to::Storage, product::Product, time::Int64, quantity::Int64)

Records that `quantity` units of `product` could not be received into `to`'s on-hand
inventory at `time` because it would have exceeded `maximum_units`, and are being held
in temporary overflow storage instead (see `get_total_overflow_costs`).
"""
function record_overflow!(state::State, to::Storage, product::Product, time::Int64, quantity::Int64)
    si = state.storage_index[to]
    pi = state.product_index[product]
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
    snapshot_state!(state::State, time, record_history::Bool)

Closes out period `time`: charges holding cost for everything currently on
hand (always, regardless of `record_history` - this is the incremental
counterpart of `get_total_holding_costs`'s history scan, see `SimMetrics`),
and, only when `record_history` is `true`, archives a per-period on-hand
snapshot plus this period's filled/placed orders into `state`'s `historical_*`
arrays for later reporting/visualization.

When `record_history` is `false` the archiving - including the Dict copy of
on-hand inventory and the per-period Set handoffs - is skipped entirely, and
`state.filled_orders`/`state.placed_orders` are just cleared in place for the
next period instead of being swapped for fresh, permanently-retained Sets.
"""
function snapshot_state!(state::State, time, record_history::Bool)
    on_hand_snapshot = record_history ? Dict{Tuple{Storage, Product}, Int64}() : nothing
    if record_history
        sizehint!(on_hand_snapshot, length(state.on_hand_totals))
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

    if record_history
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
    #println("On hand at $time, $(state.on_hand_inventory)")
end

function get_net_inventory(state::State, location::ConcreteNode, product::Product, time::Int64)
    # on-hand + in-transit + on-order from suppliers - on-order from supplied
    on_hand = get_on_hand_inventory(state, location, product)
    in_transit = sum(@view get_in_transit_inventories(state, location, product)[time:end]; init=0)
    inbound = get_inbound_orders(state, location, product, time)
    outbound = get_outbound_orders(state, location, product, time) 

    #@debug "on hand: $on_hand, in transit: $in_transit, inbound: $inbound, outbound: $outbound"

    return on_hand +
            in_transit +
            inbound -
            outbound
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
    if pi == 0
        return 0
    end
    total = 0
    for ol in state.pending_inbound_order_lines[li, pi]
        if ol.due_date >= time
            total += ol.quantity
        end
    end
    return total
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
    if pi == 0
        return 0
    end
    total = 0
    for ol in state.pending_outbound_order_lines[li, pi]
        if ol.due_date >= time
            total += ol.quantity
        end
    end
    return total
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
function get_past_outbound_orders!(past_orders::Array{Union{Missing, Int64}, 1}, state::State, location::ConcreteNode, product::Product, time::Int64)::Array{Union{Missing, Int64}, 1}
    li = get(state.location_index, location, 0)
    pi = li == 0 ? 0 : get(state.product_index, product, 0)
    for t in 1:length(past_orders)
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

function get_net_network_inventory(state, location, product)
end

function get_used_trucks(state)
    return [trip.truck for trip in state.historical_transportation]
end