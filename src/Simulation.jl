# Receive inventory
function receive_inventory!(state::State, env::Env, location::Storage, product, time)
    #println(state)
    quantity = get_in_transit_inventory(state, location, product, time)
    add_on_hand_inventory!(state, location, product, quantity)
    add_in_transit_inventory!(state, location, product, time, -quantity)
    @debug "Received at $time, $location, $product, $quantity"
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
    order_lines = collect(get(state.pending_outbound_order_lines, (location, product), OrderLine[]))
    sort!(order_lines, by=ol -> ol.due_date)
    sort!(order_lines, by=ol -> ol.creation_time)
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

function send_inventory!(state::State, env::Env, location, product::Product, time::Int)
    #println("send_inventory $location $product $time")
    if !haskey(state.pending_outbound_order_lines, (location, product))
        return
    end

    order_lines = collect(state.pending_outbound_order_lines[(location, product)])
    sort!(order_lines, by=ol -> ol.due_date)
    sort!(order_lines, by=ol -> ol.creation_time)
    #@debug order_lines

    #println("send_inventory order_lines $order_lines")
    fulfilled_order_lines = Set{OrderLine}()
    for order_line in order_lines
        if order_line.due_date < time
            push!(fulfilled_order_lines, order_line)
            continue
        end
        
        #println("send_inventory on_hand $(state.on_hand_inventory[location][order_line.product]) vs $(order_line.quantity)")
        if order_line.quantity <= state.on_hand_inventory[(location, product)]
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
            state.on_hand_inventory[(location, product)] -= order_line.quantity
            
            push!(fulfilled_order_lines, order_line)
            
            push!(state.historical_transportation, order_line.trip)

            push!(state.filled_orders, order_line)

            if state.on_hand_inventory[(location, product)] == 0
                break
            end
        end
    end

    delete_order_lines!(state, fulfilled_order_lines)
end

# Place orders
function place_orders(state::State, env::Env, location::Customer, product::Product, time::Int64)
    quantity = Int(state.demand[(location, product)].demand[time])
    if quantity > 0
        trips = env.supplying_trips[location]
        trip_index = findfirst(t -> t.departure >= time, trips)
        trip = trips[trip_index]

        order = OrderLine(time, trip.route.origin, location, product, quantity, time, missing) # customers orders are due immediately
        #@debug "Ordered at $time, $location, $product, $quantity"
        push!(state.placed_orders, order)
        
        return [order]
    else
        return OrderLine[]
    end
end
    
function place_orders(state::State, env::Env, location, product::Product, time::Int)
    orders = OrderLine[]
    for trip in get_inbound_trips(env, location, time)
        #println(policies)
        policy = get(trip.policies, product, nothing)
        if !isnothing(policy)
            quantity = get_order(policy, state, env, location, trip.route, product, time) 
            if quantity > 0
                order = OrderLine(time, trip.route.origin, location, product, quantity, typemax(Int64), trip)
                @debug "Ordered at $time, $location, $product, $quantity from $(trip.route.origin) with lead time $(trip.route.times[1])"
                
                push!(orders, order)
                push!(state.placed_orders, order)
            end
        end
    end
    return orders
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
function simulate(supplychain::SupplyChain, policies, initial_state::State)
    return simulate(Env(supplychain, [initial_state], policies), policies, initial_state)
end

"""
    simulate(env::Env, policies, initial_state::State)

    Simulates the supply chain for horizon steps, starting from the initial state.
"""
function simulate(env::Env, policies, initial_state::State)
    for storage in env.supplychain.storages
        for product in env.supplychain.products
            if get_initial_inventory(storage, product) > 0
                set_on_hand_inventory!(initial_state, storage, product, get_initial_inventory(storage, product))
            end
        end
    end

    for lane in env.supplychain.lanes
        if !isnothing(lane.initial_arrivals)
            for (product, arrivals) in lane.initial_arrivals
                for i in 1:length(lane.destinations)
                    for j in 1:length(arrivals[i])
                        add_in_transit_inventory!(initial_state, lane.destinations[i], product, j, arrivals[i][j])
                    end
                end
            end
        end
    end

    #println(initial_state)
    state = deepcopy(initial_state)
    snapshot_state!(state, 0)

    for time in 1:env.supplychain.horizon
        for location in env.sorted_locations
            for product in env.supplychain.products
                receive_inventory!(state, env, location, product, time)
            end
        end

        for location in reverse(env.sorted_locations)
            for product in env.supplychain.products
                orders = place_orders(state, env, location, product, time)
                receive_orders!(state, env, orders)
            end
        end

        for location in env.sorted_locations
            for product in env.supplychain.products
                receive_inventory!(state, env, location, product, time)
                send_inventory!(state, env, location, product, time)
            end
        end

        snapshot_state!(state, time)
    end

    return state
end
