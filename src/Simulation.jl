# Receive inventory
function receive_inventory!(state::State, env::Env, location::Storage, product, time)
    #println(state)
    quantity = get_in_transit_inventory(state, location, product, time)
    max_capacity = get_maximum_storage(location, product)

    if isinf(max_capacity)
        add_on_hand_inventory!(state, location, product, quantity, time)
        add_in_transit_inventory!(state, location, product, time, -quantity)
        return
    end

    capacity_remaining = max(0, Int(round(max_capacity)) - get_on_hand_inventory(state, location, product))
    accepted = min(quantity, capacity_remaining)
    overflow = quantity - accepted

    add_on_hand_inventory!(state, location, product, accepted, time)
    add_in_transit_inventory!(state, location, product, time, -quantity)

    if overflow > 0
        record_overflow!(state, location, product, time, overflow)
        if time < get_horizon(state)
            # excess is delayed, not lost: it waits and is retried the next period
            add_in_transit_inventory!(state, location, product, time + 1, overflow)
        end
    end
end

function receive_inventory!(state::State, env::Env, location::Customer, product, time)
    #println(state)
    quantity = get_in_transit_inventory(state, location, product, time)
    add_in_transit_inventory!(state, location, product, time, -quantity)
end

function receive_inventory!(state::State, env::Env, location::Supplier, product, time)
    #no-op
end

# Send inventory (detailed)
function send_inventory!(state::State, env::Env, trip::Trip, destination, product, quantity, time)
    #println("send_inventory_low $destination $product $quantity $time")
    if time + get_leadtime(trip.route, destination) > get_horizon(state)
        return
    end
    add_in_transit_inventory!(state, destination, product, time + get_leadtime(trip.route, destination), quantity)
end

function send_inventory!(state::State, env::Env, trip::Trip, destination::Customer, product, quantity, time)
    #no-op
end

# Metrics: fill/drop sites
"""
    record_fill!(state::State, env::Env, order_line::OrderLine)

Incrementally updates `state.metrics` for an `order_line` that has just been
fulfilled (its trip has been assigned and the shipment sent), and, if
`env.record_history` is set, mirrors the same fact into
`state.historical_transportation` for later reporting/visualization.

Must be called exactly once per fulfilled order line - matches the site of
each `push!(state.filled_orders, order_line)` in `send_inventory!`.
"""
function record_fill!(state::State, env::Env, order_line::OrderLine)
    trip = order_line.trip
    metrics = state.metrics

    metrics.trip_unit_costs += trip.route.unit_cost * order_line.quantity
    if trip ∉ metrics.seen_trips
        push!(metrics.seen_trips, trip)
        metrics.trip_fixed_costs += get_fixed_cost(trip.route)
    end
    if order_line.destination isa Customer
        metrics.sales += order_line.quantity * state.demand[(order_line.destination, order_line.product)].sales_price
    end

    if env.record_history
        push!(state.historical_transportation, trip)
    end
end

"""
    record_drop!(state::State, order_line::OrderLine)

Incrementally updates `state.metrics` for an `order_line` that has expired
(its due date passed) without being fulfilled - a lost sale, if it was bound
for a customer. Non-customer-bound order lines (e.g. internal replenishment)
never contribute to lost sales, matching `get_total_lost_sales`'s
`isa(ol.destination, Customer)` filter.

Must be called exactly once per dropped order line: either at the explicit
`due_date < time` expiry site in `send_inventory!`, or, for order lines still
pending when the horizon ends (which never reach that check - see
`flush_pending_as_lost!`), once at the end of `simulate`.
"""
function record_drop!(state::State, order_line::OrderLine)
    if order_line.destination isa Customer
        state.metrics.lost_sales += order_line.quantity * state.demand[(order_line.destination, order_line.product)].lost_sales_cost
    end
end

"""
    record_placement!(state::State, env::Env, order_line::OrderLine)

Records `order_line` into `state.outbound_order_quantities` (see
`get_past_outbound_orders`), if `env.needs_outbound_order_index` - i.e. only
if some policy in this run actually declared `required_lookback(policy) > 0`
(see `Policy.jl`). A no-op otherwise: nothing reads this index, so nothing
gets written into it (the backing arrays are always allocated regardless -
see `outbound_order_quantities`'s field doc in `State.jl` - this flag only
gates whether anything is ever written there).

Must be called exactly once per order line, at the `push!(state.placed_orders, order)`
site in each `place_orders` method.
"""
function record_placement!(state::State, env::Env, order_line::OrderLine)
    if env.needs_outbound_order_index
        li = state.location_index[order_line.origin]
        pi = state.product_index[order_line.product]
        state.outbound_order_quantities[li, pi][order_line.creation_time] += order_line.quantity
    end
end

# Send inventory
function send_inventory!(state::State, env::Env, location::Supplier, product::Product, time::Int)
    order_lines = state.pending_outbound_order_lines[state.location_index[location], state.product_index[product]]
    if isempty(order_lines)
        return
    end

    sort!(order_lines, by=ol -> (ol.creation_time, ol.due_date))
    #@debug order_lines

    fulfilled_or_dropped = OrderLine[]

    for order_line in order_lines
        if order_line.due_date < time
            record_drop!(state, order_line)
            push!(fulfilled_or_dropped, order_line)
            delete_inbound_order_line!(state, order_line)
            continue
        end

        if ismissing(order_line.trip) || order_line.trip.departure < time
            trip = find_next_departure(env, order_line.destination, time, order_line.due_date)
            if isnothing(trip)
                continue
            end
            order_line.trip = trip
        end

        send_inventory!(state, env, order_line.trip, order_line.destination, order_line.product, order_line.quantity, time)

        push!(fulfilled_or_dropped, order_line)
        delete_inbound_order_line!(state, order_line)

        record_fill!(state, env, order_line)
        push!(state.filled_orders, order_line)
    end

    if !isempty(fulfilled_or_dropped)
        filter!(ol -> ol ∉ fulfilled_or_dropped, order_lines)
    end
end

function send_inventory!(state::State, env::Env, location::ConcreteNode, product::Product, time::Int)
    #println("send_inventory $location $product $time")
    order_lines = state.pending_outbound_order_lines[state.location_index[location], state.product_index[product]]
    if isempty(order_lines)
        return
    end

    sort!(order_lines, by=ol -> (ol.creation_time, ol.due_date))
    #@debug order_lines

    #println("send_inventory order_lines $order_lines")
    fulfilled_order_lines = OrderLine[]
    # Tracked locally and kept in sync with each remove_on_hand_inventory!
    # below, instead of re-reading get_on_hand_inventory (a Dict lookup)
    # twice per order line.
    available = get_on_hand_inventory(state, location, product)
    for order_line in order_lines
        if order_line.due_date < time
            record_drop!(state, order_line)
            push!(fulfilled_order_lines, order_line)
            delete_inbound_order_line!(state, order_line)
            continue
        end

        #println("send_inventory on_hand $(get_on_hand_inventory(state, location, order_line.product) vs $(order_line.quantity)")
        if order_line.quantity <= available
            if ismissing(order_line.trip) || order_line.trip.departure < time
                trip = find_next_departure(env, order_line.destination, time, order_line.due_date)
                if isnothing(trip)
                    continue
                end
                order_line.trip = trip
            end

            send_inventory!(state, env, order_line.trip,  order_line.destination, order_line.product, order_line.quantity, time)
            remove_on_hand_inventory!(state, location, product, order_line.quantity)
            available -= order_line.quantity

            push!(fulfilled_order_lines, order_line)
            delete_inbound_order_line!(state, order_line)

            record_fill!(state, env, order_line)
            push!(state.filled_orders, order_line)

            if available == 0
                break
            end
        end
    end

    if !isempty(fulfilled_order_lines)
        filter!(ol -> ol ∉ fulfilled_order_lines, order_lines)
    end
end

# Place orders
function place_orders(state::State, env::Env, location::Customer, product::Product, time::Int64, orders::Array{OrderLine, 1})
    empty!(orders)
    demand = state.demand[(location, product)]
    quantity = Int(demand.demand[time])
    if quantity > 0
        trip = find_next_departure(env, location, time)

        order = OrderLine(time, trip.route.origin, location, product, quantity, time, missing) # customers orders are due immediately
        #@debug "Ordered at $time, $location, $product, $quantity"
        push!(orders, order)
        push!(state.placed_orders, order)
        record_placement!(state, env, order)
        state.metrics.orders += quantity
        state.metrics.demand += quantity * demand.sales_price
        return
    else
        return
    end
end
    
function place_orders(state::State, env::Env, location::ConcreteNode, product::Product, time::Int, orders::Array{OrderLine, 1})
    empty!(orders)
    for trip in get_inbound_trips(env, location, time)
        #println(policies)
        order_fn = get(trip.policies, product, nothing)
        if !isnothing(order_fn)
            quantity = Int(order_fn(state, env, location, trip.route, product, time))
            if quantity > 0
                minimum_quantity = trip.route.minimum_quantity
                if minimum_quantity > 0 && quantity < minimum_quantity
                    quantity = Int(ceil(minimum_quantity))
                end
                order = OrderLine(time, trip.route.origin, location, product, quantity, typemax(Int64), trip)

                push!(orders, order)
                push!(state.placed_orders, order)
                record_placement!(state, env, order)
                state.metrics.orders += quantity
            end
        end
    end
end

# Receive orders
"""
    receive_orders!(state::State, env::Env, orders)

    Receives the orders that have been placed.
"""
function receive_orders!(state::State, env::Env, orders)
    for order in orders
        receive_order!(state, env, order)
    end
end

function receive_order!(state::State, env::Env, order::OrderLine)
    add_order_line!(state, order)
end

# Simulate
function simulate(supplychain::SupplyChain, policies::Dict{Tuple{Lane, Product}, <:InventoryOrderingPolicy})
    initial_state = State(supplychain)
    return simulate(Env(supplychain, [initial_state], policies), policies, initial_state)
end

"""
    simulate(env::Env, policies, initial_state::State)

    Simulates the supply chain for horizon steps, starting from `initial_state`.

    `initial_state` must already reflect the desired starting condition, as produced
    by `State(supply_chain)` or by `reset!(state)`. It is simulated *in place* and
    returned rather than copied, so callers that want to run multiple simulations
    from the same starting point (e.g. `optimize!` evaluating many policy
    candidates) can reuse the same `State` via `reset!` instead of paying for a
    `deepcopy` of the whole (read-only) supply chain network on every evaluation.
"""
function simulate(env::Env, policies, initial_state)
    orders = OrderLine[]

    state = initial_state
    snapshot_state!(state, 0, env.record_history)

    # env.sorted_locations never changes across periods, so reverse it once
    # instead of allocating a fresh reversed copy every period.
    reversed_sorted_locations = reverse(env.sorted_locations)

    for time in 1:env.supplychain.horizon
        for location in env.sorted_locations
            for product in env.supplychain.products
                receive_inventory!(state, env, location, product, time)
            end
        end

        for location in reversed_sorted_locations
            for product in env.supplychain.products
                place_orders(state, env, location, product, time, orders)
                receive_orders!(state, env, orders)
            end
        end

        for location in env.sorted_locations
            for product in env.supplychain.products
                receive_inventory!(state, env, location, product, time)
                send_inventory!(state, env, location, product, time)
            end
        end

        for location in env.sorted_locations
            if isa(location, Storage)
                for product in env.supplychain.products
                    expire_on_hand_inventory(state, location, product, time)
                end
            end
        end

        snapshot_state!(state, time, env.record_history)
    end

    flush_pending_as_lost!(state)

    return state
end

"""
    flush_pending_as_lost!(state::State)

A customer order line's `due_date` equals the period it was created in (see
`place_orders(..., ::Customer, ...)`), and the explicit drop site in
`send_inventory!` only recognizes it as expired - and calls `record_drop!` -
once `due_date < time`. An order line created in the very last period of the
horizon never sees a later period, so that check never fires for it: it
simply stays in `pending_outbound_order_lines` forever.

`get_total_lost_sales` doesn't notice this gap because it isn't
event-driven: it diffs the set of every order ever placed against the set of
every order ever filled, so a never-filled last-period order line shows up
as lost regardless of whether anything explicitly marked it as dropped.
`state.metrics.lost_sales`, being event-driven, needs this equivalent
one-time sweep over whatever is still pending when the horizon ends, so it
accounts for exactly the same order lines - no more, no less: anything
already recorded via the due_date < time path was already deleted from
pending_outbound_order_lines, so there is no overlap/double-count between
that path and this one.
"""
function flush_pending_as_lost!(state::State)
    for order_line in Base.Iterators.flatten(values(state.pending_outbound_order_lines))
        record_drop!(state, order_line)
    end
end
