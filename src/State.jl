import Base.push!
import Base.delete!

"""
Contains information about the historical and current state of the simulation, including inventory positions and pending orders.
"""
mutable struct State
    supply_chain::SupplyChain
    demand::Dict{Tuple{Customer, Product}, Demand}

    on_hand_inventory::Dict{Tuple{Storage, Product}, Dict{Int64, Int64}}

    in_transit_inventory::Dict{Tuple{<:Node, Product}, Array{Int64, 1}}

    overflow_inventory::Dict{Tuple{Storage, Product}, Array{Int64, 1}}

    pending_outbound_order_lines::Dict{Tuple{<:Node, Product}, Set{OrderLine}}
    pending_inbound_order_lines::Dict{Tuple{<:Node, Product}, Set{OrderLine}}

    filled_orders::Set{OrderLine}
    placed_orders::Set{OrderLine}

    historical_on_hand::Array{Dict{Tuple{Storage, Product}, Int64}, 1}
    historical_orders::Array{Set{OrderLine}, 1}
    historical_transportation::Set{Trip}
    historical_filled_orders::Array{Set{OrderLine}, 1}
    #historical_pending_outbound_order_lines::Array{Dict{Node, Set{OrderLine}}}

    # Incrementally-updated running totals, kept in sync with the
    # historical_* arrays above regardless of Env.record_history - see
    # SimMetrics.
    metrics::SimMetrics

    # Quantity ordered per (origin, product, creation_time), maintained only
    # when Env.needs_outbound_order_index is set (i.e. some policy actually
    # declares required_lookback(policy) > 0 - see Policy.jl). Lets
    # get_past_outbound_orders answer "how much did this location ship out
    # at period t" with a direct array read instead of rescanning
    # historical_orders' per-period Sets (which hold every order placed by
    # every location, not just the one being asked about) for a match.
    outbound_order_quantities::Dict{Tuple{<:Node, Product}, Vector{Int64}}

    function State(supply_chain; pending_outbound_order_lines=Dict{Storage, Array{OrderLine, 1}}())
        demand = Dict((d.customer, d.product) => d for d in supply_chain.demand)

        state = new(supply_chain,
                   demand,
                   Dict{Tuple{Storage, Product}, Int64}(),
                   Dict{Tuple{<:Node, Product}, Array{Int64, 1}}(),
                   Dict{Tuple{Storage, Product}, Array{Int64, 1}}(),
                   Dict{Tuple{<:Node, Product}, Set{OrderLine}}(),
                   Dict{Tuple{<:Node, Product}, Set{OrderLine}}(),
                   Set{OrderLine}(),
                   Set{OrderLine}(),
                   [],
                   OrderLine[],
                   Set{Trip}(),
                   [],
                   SimMetrics(),
                   Dict{Tuple{<:Node, Product}, Vector{Int64}}())
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
    empty!(state.on_hand_inventory)
    empty!(state.in_transit_inventory)
    empty!(state.overflow_inventory)
    empty!(state.pending_outbound_order_lines)
    empty!(state.pending_inbound_order_lines)
    empty!(state.filled_orders)
    empty!(state.placed_orders)
    empty!(state.historical_on_hand)
    empty!(state.historical_orders)
    empty!(state.historical_transportation)
    empty!(state.historical_filled_orders)
    reset!(state.metrics)
    empty!(state.outbound_order_quantities)

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
    t1 = (order_line.origin, order_line.product)
    if !haskey(state.pending_outbound_order_lines, t1) 
        state.pending_outbound_order_lines[t1] = Set{OrderLine}()
    end
    t2 = (order_line.destination, order_line.product)
    if !haskey(state.pending_inbound_order_lines, t2) 
        state.pending_inbound_order_lines[t2] = Set{OrderLine}()
    end

    Base.push!(state.pending_outbound_order_lines[t1], order_line)
    Base.push!(state.pending_inbound_order_lines[t2], order_line)
end

function delete_order_line!(state::State, order_line::OrderLine)
    Base.delete!(state.pending_outbound_order_lines[(order_line.origin, order_line.product)], order_line)
    Base.delete!(state.pending_inbound_order_lines[(order_line.destination, order_line.product)], order_line)
end

function delete_order_lines!(state::State, order_lines)
    for order_line in order_lines
        delete_order_line!(state, order_line)
    end
end

function set_on_hand_inventory!(state::State, to::Node, product::Product, quantity, time)
    if !haskey(state.on_hand_inventory, (to, product))
        state.on_hand_inventory[(to, product)] = Dict{Int64, Int64}()
    end
    state.on_hand_inventory[(to, product)][time] = Int(quantity)
end

function add_on_hand_inventory!(state::State, to::Storage, product::Product, quantity::Int64, time)
    if !haskey(state.on_hand_inventory, (to, product))
        state.on_hand_inventory[(to, product)] = Dict{Int64, Int64}()
    end
    state.on_hand_inventory[(to, product)][time] = get(state.on_hand_inventory[(to, product)], time, 0) + quantity
end

function remove_on_hand_inventory!(state::State, to::Storage, product::Product, quantity::Int64)
    ages = state.on_hand_inventory[(to, product)]
    # FIFO: must consume oldest inventory (smallest age/arrival time) first, so
    # this ordering is required, not just habit - sort! (in place) instead of
    # sort avoids the extra defensive copy sort() makes of its input.
    for t in sort!(collect(keys(ages)))
        if quantity <= 0
            break
        end
        removed_quantity = min(quantity, ages[t])
        ages[t] -= removed_quantity
        quantity -= removed_quantity
    end
end

function get_on_hand_inventory(state::State, to::Node, product::Product)::Int64
    return sum(values(get(state.on_hand_inventory, (to, product), Dict{Int64, Int64}())); init=0)
end

function expire_on_hand_inventory(state::State, to::Storage, product::Product, time)
    max_age = get_maximum_age(to, product)
    ages = get(state.on_hand_inventory, (to, product), nothing)
    if isnothing(ages)
        return
    end
    # Order doesn't matter here (unlike remove_on_hand_inventory!'s FIFO
    # consumption): every bucket past max_age gets zeroed regardless of which
    # order they're visited in, so no need to collect+sort the keys.
    for (t, on_hand) in ages
        if (t <= time - max_age) && (on_hand > 0)
            ages[t] = 0
        end
    end
    return
end

function add_in_transit_inventory!(state::State, to::N, product::Product, time::Int64, quantity::Int64) where N <: Node
    if !haskey(state.in_transit_inventory, (to, product))
        # zeros(n) (no explicit type) allocates a Vector{Float64}, requiring
        # an implicit convert to the field's declared Array{Int64, 1} on
        # assignment below - one that happens to succeed today only because
        # every entry starts out an exact 0.0, and silently depends on that
        # continuing to hold. Allocating the right element type directly
        # avoids relying on that at all.
        state.in_transit_inventory[(to, product)] = zeros(Int64, get_horizon(state))
    end
    state.in_transit_inventory[(to, product)][time] += quantity
end

function delete_in_transit_inventory!(state::State, to::N, product::Product, time::Int64, quantity::Int64) where N <: Node
    state.in_transit_inventory[(to, product)][time] -= quantity
end

"""
    get_in_transit_inventory(state::State, to::Location, product::Product, time::Int64)::Int64

    Gets the number of units of a product in transit to a location at a given time.
"""
function get_in_transit_inventory(state::State, to::N, product::Product, time::Int64)::Int64 where N <: Node
    in_transit_inventory = get(state.in_transit_inventory, (to, product), nothing)
    if isnothing(in_transit_inventory)
        return 0
    end
    return in_transit_inventory[time]
end

function get_in_transit_inventories(state::State, to::N, product::Product)::Array{Int64, 1} where N <: Node
    if !haskey(state.in_transit_inventory, (to, product))
        return [0]
    end
    return state.in_transit_inventory[(to, product)]
end

"""
    record_overflow!(state::State, to::Storage, product::Product, time::Int64, quantity::Int64)

Records that `quantity` units of `product` could not be received into `to`'s on-hand
inventory at `time` because it would have exceeded `maximum_units`, and are being held
in temporary overflow storage instead (see `get_total_overflow_costs`).
"""
function record_overflow!(state::State, to::Storage, product::Product, time::Int64, quantity::Int64)
    if !haskey(state.overflow_inventory, (to, product))
        state.overflow_inventory[(to, product)] = zeros(Int64, get_horizon(state))
    end
    # This slot is overwritten, not accumulated (receive_inventory! can call
    # record_overflow! more than once for the same (to, product, time) in a
    # period), so the metrics update must track the delta rather than adding
    # the new quantity outright - otherwise a corrected/overwritten value
    # would be double-counted relative to get_total_overflow_costs, which
    # only ever sees the final value left in the slot.
    previous_quantity = state.overflow_inventory[(to, product)][time]
    state.overflow_inventory[(to, product)][time] = quantity
    state.metrics.overflow_costs += (quantity - previous_quantity) * get_overflow_cost(to, product)
end

"""
    get_overflow_inventory(state::State, to::Storage, product::Product, time::Int64)::Int64

Gets the number of units of `product` held in temporary overflow storage at `to` at `time`.
"""
function get_overflow_inventory(state::State, to::Storage, product::Product, time::Int64)::Int64
    overflow = get(state.overflow_inventory, (to, product), nothing)
    if isnothing(overflow)
        return 0
    end
    return overflow[time]
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
        sizehint!(on_hand_snapshot, length(state.on_hand_inventory))
    end
    for (k, ages) in state.on_hand_inventory
        (location, product) = k
        on_hand = sum(values(ages); init=0)
        if on_hand != 0
            state.metrics.holding_costs += on_hand * get(location.unit_holding_cost, product, 0)
        end
        if record_history
            on_hand_snapshot[k] = on_hand
        end
    end

    if record_history
        push!(state.historical_on_hand, on_hand_snapshot)

        # state.filled_orders/placed_orders are only ever mutated via push! on the
        # field itself, so handing the current Set to history and replacing the
        # field with a fresh one is equivalent to copy+empty! without the O(n)
        # element-by-element copy.
        push!(state.historical_filled_orders, state.filled_orders)
        state.filled_orders = Set{OrderLine}()

        push!(state.historical_orders, state.placed_orders)
        state.placed_orders = Set{OrderLine}()
    else
        empty!(state.filled_orders)
        empty!(state.placed_orders)
    end
    #push!(state.historical_pending_outbound_order_lines, Dict(k => copy(v) for (k, v) in state.order_line_tracker.pending_inbound_order_lines))
    #println("On hand at $time, $(state.on_hand_inventory)")
end

function get_net_inventory(state::State, location::Node, product::Product, time::Int64)
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
function get_inbound_orders(state::State, location::Node, product::Product, time::Int64)::Int64
    sum(ol -> (ol.due_date >= time) ? ol.quantity : 0, 
        get(state.pending_inbound_order_lines, (location, product), OrderLine[]);
        init = 0
    )
end

"""
    get_outbound_orders(state::State, location::Location, product::Product, time::Int64)::Int64

    Gets the number of units of a product on order at a location (and not yet shipped out) at a given time.
"""
function get_outbound_orders(state::State, location::Node, product::Product, time::Int64)::Int64
    sum(ol -> (ol.due_date >= time) ? ol.quantity : 0, 
        get(state.pending_outbound_order_lines, (location, product), OrderLine[]);
        init = 0
    )
end

"""
    get_past_outbound_orders(state::State, location::Node, product::Product, time::Int64, step_back::Int64)::Array{Union{Missing, Int64}, 1}

Gets, for each of the `step_back` periods before `time`, the quantity of
`product` that was ordered *from* `location` (i.e. `location` was the
`origin`) - `missing` for any period before the simulation started.

Reads `state.outbound_order_quantities`, which is only populated when some
policy declares `required_lookback(policy) > 0` (see `Policy.jl`/`Env`); for
a `location`/`product` nothing was ever recorded for (either because no such
policy is in play, or simply because no order ever originated there), every
period reads back as `0`, matching what an exhaustive scan would have found.
"""
function get_past_outbound_orders(state::State, location::Node, product::Product, time::Int64, step_back::Int64)::Array{Union{Missing, Int64}, 1}
    past_orders = zeros(Union{Missing, Int64}, step_back)
    quantities = get(state.outbound_order_quantities, (location, product), nothing)
    for t in 1:step_back
        creation_time = time - t
        if creation_time < 0
            past_orders[t] = missing
        elseif creation_time == 0 || isnothing(quantities)
            # creation_time == 0 predates period 1 (nothing is ever placed
            # then), same as the pre-simulation snapshot the old
            # historical_orders-scanning implementation read as empty here -
            # not `missing`, unlike creation_time < 0 above.
            past_orders[t] = 0
        else
            past_orders[t] = quantities[creation_time]
        end
    end
    past_orders
end

function get_net_network_inventory(state, location, product)
end

function get_used_trucks(state)
    return [trip.truck for trip in state.historical_transportation]
end