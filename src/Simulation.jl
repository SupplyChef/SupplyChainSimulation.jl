# Receive inventory
function receive_inventory!(state, env, location::Storage, product, time)
    #println(state)
    quantity = get_in_transit_inventory(state, location, product, time)
    state.on_hand_inventory[location][product] += quantity
    add_in_transit_inventory!(state, location, product, time, -quantity)
    if quantity > 0
        #println("Received at $time, $location, $product, $quantity")
    end
end

function receive_inventory!(state, env, location::Customer, product, time)
    #println(state)
    quantity = get_in_transit_inventory(state, location, product, time)
    add_in_transit_inventory!(state, location, product, time, -quantity)
    if quantity > 0
        println("receive_inventory $location $product $quantity $time")
        #println("Received at $time, $location, $product, $quantity")
    end
end

function receive_inventory!(state, env, location::Supplier, product, time)
    #no-op
end

# Send inventory (detailed)
function send_inventory!(state, env, trip::Trip, destination, product, quantity, time)
    #println("send_inventory_low $destination $product $quantity $time")
    if time + get_leadtime(trip.route, destination) > get_horizon(state)
        return
    end
    add_in_transit_inventory!(state, destination, product, time + get_leadtime(trip.route, destination), quantity)
    #println("Sent at $time, $(trip.route.origin), $destination, $product, $quantity")
end

function send_inventory!(state, env, trip::Trip, destination::Customer, product, quantity, time)
    #no-op
end

# Send inventory
function send_inventory!(state::State, env::Env, location::Supplier, product::Product, time::Int)
    order_lines = sort(collect(filter(ol -> ol.product == product, state.order_line_tracker.pending_outbound_order_lines[location])), by=ol -> ol.order.due_date)
    
    for order_line in order_lines
        if order_line.order.due_date < time
            delete_order_line!(state, order_line)
            continue
        end

        # if true
            trip = first(filter(t -> t.departure >= time, env.supplying_trips[order_line.order.destination]))

            order_line.trip = trip
            send_inventory!(state, env, trip, order_line.order.destination, order_line.product, order_line.quantity, time)
            
            delete_order_line!(state, order_line)
            
            if !haskey(state.historical_transportation, order_line.trip)
                state.historical_transportation[order_line.trip] = OrderLine[]
            end
            push!(state.historical_transportation[order_line.trip], order_line)
            push!(state.filled_order_lines, order_line)
        # end
    end
end

function send_inventory!(state::State, env::Env, location, product::Product, time::Int)
    #println("send_inventory $location $product $time")
    if !haskey(state.order_line_tracker.pending_outbound_order_lines, location)
        return
    end

    order_lines = sort(collect(filter(ol -> ol.product == product, state.order_line_tracker.pending_outbound_order_lines[location])), by=ol -> ol.order.due_date)

    #println("send_inventory order_lines $order_lines")
    fulfilled_order_lines = Set{OrderLine}()
    for order_line in order_lines
        if order_line.order.due_date < time
            push!(fulfilled_order_lines, order_line)
            continue
        end
        
        #println("send_inventory on_hand $(state.on_hand_inventory[location][order_line.product]) vs $(order_line.quantity)")
        if order_line.quantity <= state.on_hand_inventory[location][product]
            trip = first(filter(t -> t.departure >= time, env.supplying_trips[order_line.order.destination]))

            order_line.trip = trip
            send_inventory!(state, env, trip,  order_line.order.destination, order_line.product, order_line.quantity, time)
            state.on_hand_inventory[location][product] -= order_line.quantity
            
            push!(fulfilled_order_lines, order_line)
            
            if !haskey(state.historical_transportation, order_line.trip)
                state.historical_transportation[order_line.trip] = OrderLine[]
            end
            push!(state.historical_transportation[order_line.trip], order_line)

            push!(state.filled_order_lines, order_line)

            if state.on_hand_inventory[location][product] == 0
                break
            end
        end
    end

    delete_order_lines!(state, fulfilled_order_lines)
end

# Place orders
function place_orders(state::State, env::Env, location::Customer, product::Product, time::Int64)
    quantity = state.demand[(location, product)][time]
    if quantity > 0
        trip = first(filter(t -> t.departure >= time, env.supplying_trips[location]))
        order = Order(trip.route.origin, location, Set{OrderLine}(), time) # customers orders are due immediately
        push!(order.lines, OrderLine(order, product, quantity))
        
        orders = [order]
        #println("Place $order")
        
        push!(state.historical_orders, order)
        
        return orders
    else
        return Order[]
    end
end
    
function place_orders(state::State, env::Env, location, product::Product, time::Int)
    orders = Order[]
    for trip in get_inbound_trips(env, location, time)
        #println(state.policies)
        policy = state.policies[(trip.route, product)]
        quantity = get_order(policy, state, env, location, trip.route, product, time) 
        if quantity > 0
            order = Order(trip.route.origin, location, Set{OrderLine}(), typemax(Int64)) # internal orders are backlogged
            push!(order.lines, OrderLine(order, product, quantity))
            
            push!(orders, order)
            #println("Place $order")
            
            push!(state.historical_orders, order)
        end
    end
    return orders
end

# Receive orders
function receive_orders!(state::State, env::Env, orders)
    for order in orders
        receive_order!(state, env, order)
    end
end

function receive_order!(state, env, order)
    for order_line in order.lines
        add_order_line!(state, order_line)
    end
end

# Simulate
function simulate(network::Network, horizon::Int64, initial_state::State)
    return simulate(Env(network, [initial_state]), horizon, initial_state)
end

function simulate(env::Env, horizon::Int64, initial_state::State)
    state = deepcopy(initial_state)
    snapshot_state!(state, 0)

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
            end
        end

        for location in sorted_locations
            for product in env.network.products
                receive_inventory!(state, env, location, product, time)
                send_inventory!(state, env, location, product, time)
            end
        end

        snapshot_state!(state, time)
    end

    return state
end
