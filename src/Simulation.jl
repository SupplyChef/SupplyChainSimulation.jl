# Receive inventory
function receive_inventory!(state, env, location::Storage, product, time)
    #println(state)
    quantity = state.in_transit_inventory[location][product][time]
    state.on_hand_inventory[location][product] += quantity
    state.in_transit_inventory[location][product][time] = 0
    if quantity > 0
        #println("Received at $time, $location, $product, $quantity")
    end
end

function receive_inventory!(state, env, location::Customer, product, time)
    #println(state)
    quantity = state.in_transit_inventory[location][product][time]
    state.in_transit_inventory[location][product][time] = 0
    if quantity > 0
        #println("Received at $time, $location, $product, $quantity")
    end
end

function receive_inventory!(state, env, location::Supplier, product, time)
    #no-op
end

# Send inventory (detailed)
function send_inventory!(state, env, trip::Trip, destination, product, quantity, time)
    if time + get_leadtime(trip.route, destination) > length(state.in_transit_inventory[destination][product])
        return
    end
    state.in_transit_inventory[destination][product][time + get_leadtime(trip.route, destination)] += quantity
    #println("Sent at $time, $(trip.route.origin), $destination, $product, $quantity")
end

function send_inventory!(state, env, trip::Trip, destination::Customer, product, quantity, time)
    #no-op
end

# Send inventory
function send_inventory!(state, env, location::Supplier, product, time)
    order_lines = sort(filter(ol -> ol.order.trip.route.origin == location 
                                    && ol.order.due_date >= time
                                    && ol.product == product, state.pending_order_lines[location]), by=ol -> ol.order.due_date)
    for order_line in order_lines
        # if true
            send_inventory!(state, env, order_line.order.trip, order_line.order.destination, order_line.product, order_line.quantity, time)
            
            filter!(ol -> ol != order_line, state.pending_order_lines[location])
            
            if !haskey(state.historical_transportation, order_line.order.trip)
                state.historical_transportation[order_line.order.trip] = OrderLine[]
            end
            push!(state.historical_transportation[order_line.order.trip], order_line)
            push!(state.historical_lines_filled, order_line)
        # end
    end
end

function send_inventory!(state, env, location, product, time)
    if !haskey(state.pending_order_lines, location)
        return
    end

    order_lines = sort(filter(ol -> ol.order.trip.route.origin == location 
                                    && ol.order.due_date >= time
                                    && ol.product == product, state.pending_order_lines[location]), by=ol -> ol.order.due_date)

    fulfilled_order_lines = Set{OrderLine}()
    for order_line in order_lines
        if order_line.quantity <= state.on_hand_inventory[location][order_line.product]
            send_inventory!(state, env, order_line.order.trip,  order_line.order.destination, order_line.product, order_line.quantity, time)
            state.on_hand_inventory[location][product] -= order_line.quantity
            
            push!(fulfilled_order_lines, order_line)
            
            if !haskey(state.historical_transportation, order_line.order.trip)
                state.historical_transportation[order_line.order.trip] = OrderLine[]
            end
            push!(state.historical_transportation[order_line.order.trip], order_line)

            push!(state.historical_lines_filled, order_line)

            if state.on_hand_inventory[location][product] == 0
                break
            end
        end
    end

    filter!(ol -> ol ∉ fulfilled_order_lines, state.pending_order_lines[location])
end

# Place orders
function place_orders(state, env, location::Customer, product, time)
    quantity = state.demand[(location, product)][time]
    if quantity > 0
        trip = first(filter(t -> is_destination(location, t.route) && t.departure >= time, env.supplying_trips[location]))
        order = Order(location, trip, OrderLine[], time)
        push!(order.lines, OrderLine(order, product, quantity))
        
        orders = [order]
        #println("Place $order")
        
        push!(state.historical_orders, order)
        
        return orders
    else
        return Order[]
    end
end
    
function place_orders(state, env, location, product, time)
    orders = Order[]
    for trip in get_inbound_trips(env, location, time)
        #println(state.policies)
        policy = state.policies[(trip.route, product)]
        quantity = get_order(policy, state, env, location, trip.route, product, time)
        if quantity > 0
            order = Order(location, trip, OrderLine[], time)
            push!(order.lines, OrderLine(order, product, quantity))
            
            push!(orders, order)
            #println("Place $order")
            
            push!(state.historical_orders, order)
        end
    end
    return orders
end

# Receive orders
function receive_orders!(state, env, orders)
    for order in orders
        receive_order!(state, env, order)
    end
end

function receive_order!(state, env, order)
    for order_line in order.lines
        push!(state.pending_order_lines[order.trip.route.origin], order_line)
    end
end

# Simulate
function simulate(network::Network, horizon::Int64, initial_state::State)
    return simulate(Env(network, [initial_state]), horizon, initial_state)
end

function simulate(env::Env, horizon::Int64, initial_state::State)
    state = deepcopy(initial_state)

    sorted_locations = get_sorted_locations(env.network)

    for time in 1:horizon
        for location in sorted_locations
            for product in env.network.products
                receive_inventory!(state, env, location, product, time)
            end
        end

        for location in reverse(sorted_locations)
            for product in env.network.products
                orders = place_orders(state, env, location, product, time)
                #println("Orders $location, $product: $orders")
                receive_orders!(state, env, orders)
                send_inventory!(state, env, location, product, time)
            end
        end

        for location in sorted_locations
            for product in env.network.products
                receive_inventory!(state, env, location, product, time)
            end
        end

        snapshot_state!(state, time)
    end

    return state
end
