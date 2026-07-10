# Receive inventory
function receive_inventory!(state::State, env::Env, location::Storage, product, time)
    #println(state)
    quantity = get_in_transit_inventory(state, location, product, time)
    max_capacity = get_maximum_storage(location, product)

    if isinf(max_capacity)
        add_on_hand_inventory!(state, location, product, quantity, time)
        add_in_transit_inventory!(state, location, product, time, -quantity)
        @debug "Received at $time, $location, $product, $quantity"
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
        @debug "Capacity exceeded at $time, $location, $product: accepted $accepted, overflowed $overflow"
    end

    @debug "Received at $time, $location, $product, $accepted"
end

function receive_inventory!(state::State, env::Env, location::Customer, product, time)
    #println(state)
    quantity = get_in_transit_inventory(state, location, product, time)
    add_in_transit_inventory!(state, location, product, time, -quantity)
    @debug "Received at $time, $location, $product, $quantity"
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
    @debug "Sent at $time, $(trip.route.origin), $destination, $product, $quantity with lead time $(trip.route.times[1])"
end

function send_inventory!(state::State, env::Env, trip::Trip, destination::Customer, product, quantity, time)
    #no-op
    @debug "Sent at $time, $(trip.route.origin), $destination, $product, $quantity with lead time $(trip.route.times[1])"
end

# Send inventory
function send_inventory!(state::State, env::Env, location::Supplier, product::Product, time::Int)
    if !haskey(state.pending_outbound_order_lines, (location, product))
        return
    end

    order_lines = collect(state.pending_outbound_order_lines[(location, product)])
    sort!(order_lines, by=ol -> (ol.creation_time, ol.due_date))
    #@debug order_lines

    for order_line in order_lines
        if order_line.due_date < time
            delete_order_line!(state, order_line)
            continue
        end

        if ismissing(order_line.trip) || order_line.trip.departure < time
            @debug "replacing trip $(order_line.trip)"
            trips = env.supplying_trips[order_line.destination]
            trip_index = findfirst(t -> (t.departure >= time) && (t.departure + t.route.times[1] <= order_line.due_date), trips)
            if isnothing(trip_index) 
                continue
            end
            trip = trips[trip_index]
            order_line.trip = trip
        end

        send_inventory!(state, env, order_line.trip, order_line.destination, order_line.product, order_line.quantity, time)
            
        delete_order_line!(state, order_line)
            
        push!(state.historical_transportation, order_line.trip)
        push!(state.filled_orders, order_line)
    end
end

function send_inventory!(state::State, env::Env, location::Node, product::Product, time::Int)
    #println("send_inventory $location $product $time")
    if !haskey(state.pending_outbound_order_lines, (location, product))
        return
    end

    order_lines = collect(state.pending_outbound_order_lines[(location, product)])
    sort!(order_lines, by=ol -> (ol.creation_time, ol.due_date))
    #@debug order_lines

    #println("send_inventory order_lines $order_lines")
    fulfilled_order_lines = OrderLine[]
    for order_line in order_lines
        if order_line.due_date < time
            push!(fulfilled_order_lines, order_line)
            continue
        end
        
        #println("send_inventory on_hand $(get_on_hand_inventory(state, location, order_line.product) vs $(order_line.quantity)")
        if order_line.quantity <= get_on_hand_inventory(state, location, product)
            if ismissing(order_line.trip) || order_line.trip.departure < time
                @debug "replacing trip $(order_line.trip)"
                trips = env.supplying_trips[order_line.destination]
                trip_index = findfirst(t -> (t.departure >= time) && (t.departure + t.route.times[1] <= order_line.due_date), trips)
                if isnothing(trip_index) 
                    continue
                end
                trip = trips[trip_index]
                order_line.trip = trip
            end

            send_inventory!(state, env, order_line.trip,  order_line.destination, order_line.product, order_line.quantity, time)
            remove_on_hand_inventory!(state, location, product, order_line.quantity)
            
            push!(fulfilled_order_lines, order_line)
            
            push!(state.historical_transportation, order_line.trip)

            push!(state.filled_orders, order_line)

            if get_on_hand_inventory(state, location, product) == 0
                break
            end
        end
    end

    delete_order_lines!(state, fulfilled_order_lines)
end

# Place orders
function place_orders(state::State, env::Env, location::Customer, product::Product, time::Int64, orders::Array{OrderLine, 1})
    empty!(orders)
    quantity = Int(state.demand[(location, product)].demand[time])
    if quantity > 0
        trips = env.supplying_trips[location]
        trip_index = findfirst(t -> t.departure >= time, trips)
        trip = trips[trip_index]

        order = OrderLine(time, trip.route.origin, location, product, quantity, time, missing) # customers orders are due immediately
        #@debug "Ordered at $time, $location, $product, $quantity"
        push!(orders, order)
        push!(state.placed_orders, order)
        return
    else
        return
    end
end
    
function place_orders(state::State, env::Env, location, product::Product, time::Int, orders::Array{OrderLine, 1})
    empty!(orders)
    for trip in get_inbound_trips(env, location, time)
        #println(policies)
        policy = get(trip.policies, product, nothing)
        if !isnothing(policy)
            quantity = Int(get_order(policy, state, env, location, trip.route, product, time))
            if quantity > 0
                minimum_quantity = trip.route.minimum_quantity
                if minimum_quantity > 0 && quantity < minimum_quantity
                    quantity = Int(ceil(minimum_quantity))
                end
                order = OrderLine(time, trip.route.origin, location, product, quantity, typemax(Int64), trip)
                @debug "Ordered at $time, $location, $product, $quantity from $(trip.route.origin) with lead time $(trip.route.times[1])"
                
                push!(orders, order)
                push!(state.placed_orders, order)
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
    snapshot_state!(state, 0)

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

        snapshot_state!(state, time)
    end

    return state
end
