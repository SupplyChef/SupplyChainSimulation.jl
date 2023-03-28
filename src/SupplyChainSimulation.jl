module SupplyChainSimulation

export Route
export Lane
export Storage
export Customer
export Supplier
export Network
export Product
export Single
export Bundle
export Order
export OrderLine

export State

export OnHandUptoOrderingPolicy
export NetUptoOrderingPolicy

export set_parameter!
export get_sorted_locations
export get_total_demand
export get_total_sales
export get_total_lost_sales
export simulate
export optimize!
export get_inbound_orders
export get_outbound_orders
export get_net_inventory
export get_holding_costs
export get_transportation_costs

export get_trips

export eoq_quantity

using Graphs
using Optim
using BlackBoxOptim

include("Model.jl")
include("State.jl")
include("Policy.jl")

include("Optimization.jl")
include("Reporting.jl")

struct SimulationResults end

# Receive inventory
function receive_inventory!(state, network, location::Storage, product, time)
    #println(state)
    quantity = state.in_transit_inventory[location][product][time]
    state.on_hand_inventory[location][product] += quantity
    state.in_transit_inventory[location][product][time] = 0
    if quantity > 0
        #println("Received at $time, $location, $product, $quantity")
    end
end

function receive_inventory!(state, network, location::Customer, product, time)
    #println(state)
    quantity = state.in_transit_inventory[location][product][time]
    state.in_transit_inventory[location][product][time] = 0
    if quantity > 0
        #println("Received at $time, $location, $product, $quantity")
    end
end

function receive_inventory!(state, network, location::Supplier, product, time)
    #no-op
end

# Send inventory (detailed)
function send_inventory!(state, network, trip::Trip, destination, product, quantity, time)
    state.in_transit_inventory[destination][product][time + get_leadtime(trip.route, destination)] += quantity
    #println("Sent at $time, $(trip.route.origin), $destination, $product, $quantity")
end

function send_inventory!(state, network, trip::Trip, destination::Customer, product, quantity, time)
    #no-op
end

# Send inventory
function send_inventory!(state, network, location::Supplier, product, time)
    order_lines = sort(filter(ol -> ol.order.trip.route.origin == location 
                                    && ol.order.due_date >= time
                                    && ol.product == product, state.pending_order_lines[location]), by=ol -> ol.order.due_date)
    for order_line in order_lines
        # if true
            send_inventory!(state, network, order_line.order.trip, order_line.order.destination, order_line.product, order_line.quantity, time)
            
            filter!(ol -> ol != order_line, state.pending_order_lines[location])
            
            
            if !haskey(state.historical_transportation, order_line.order.trip)
                state.historical_transportation[order_line.order.trip] = OrderLine[]
            end
            push!(state.historical_transportation[order_line.order.trip], order_line)
            push!(state.historical_lines_filled, order_line)
        # end
    end
end

function send_inventory!(state, network, location, product, time)
    if !haskey(state.pending_order_lines, location)
        return
    end

    order_lines = sort(filter(ol -> ol.order.trip.route.origin == location 
                                    && ol.order.due_date >= time
                                    && ol.product == product, state.pending_order_lines[location]), by=ol -> ol.order.due_date)
    for order_line in order_lines
        if order_line.quantity <= state.on_hand_inventory[location][order_line.product]
            send_inventory!(state, network, order_line.order.trip,  order_line.order.destination, order_line.product, order_line.quantity, time)
            state.on_hand_inventory[location][product] -= order_line.quantity
            
            filter!(ol -> ol != order_line, state.pending_order_lines[location])
            
            if !haskey(state.historical_transportation, order_line.order.trip)
                state.historical_transportation[order_line.order.trip] = OrderLine[]
            end
            push!(state.historical_transportation[order_line.order.trip], order_line)

            push!(state.historical_lines_filled, order_line)
        end
    end
end

# Place orders
function place_orders(state, network, location::Customer, product, time)
    quantity = state.demand[(location, product)][time]
    if quantity > 0
        trip = collect(filter(t -> location ∈ get_destinations(t.route) && t.departure >= time, network.trips))[1]
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
    
function place_orders(state, network, location, product, time)
    orders = Order[]
    for trip in get_inbound_trips(network, location, time)
        #println(state.policies)
        policy = state.policies[(trip.route, product)]
        quantity = get_order(policy, state, network, location, trip.route, product, time)
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
function receive_orders!(state, network, orders)
    for order in orders
        receive_order!(state, network, order)
    end
end

function receive_order!(state, network, order)
    for order_line in order.lines
        push!(state.pending_order_lines[order.trip.route.origin], order_line)
    end
end

# Simulate
function simulate(network::Network, horizon::Int64, initial_state::State)
    state = deepcopy(initial_state)

    sorted_locations = get_sorted_locations(network)

    for time in 1:horizon
        for location in sorted_locations
            for product in network.products
                receive_inventory!(state, network, location, product, time)
            end
        end

        for location in reverse(sorted_locations)
            for product in network.products
                orders = place_orders(state, network, location, product, time)
                #println("Orders $location, $product: $orders")
                receive_orders!(state, network, orders)
                send_inventory!(state, network, location, product, time)
            end
        end

        for location in get_sorted_locations(network)
            for product in network.products
                receive_inventory!(state, network, location, product, time)
            end
        end

        snapshot_state!(state, time)
    end

    return state
end

# EOQ
"""
    eoq_quantity(demand_rate, ordering_cost, holding_cost_rate)

    Computes the economic ordering quantity that minimizes overall costs (ordering costs + holding costs) while meeting  demand.
"""
function eoq_quantity(demand_rate, ordering_cost, holding_cost_rate)
    return sqrt((2 * demand_rate * ordering_cost) / (holding_cost_rate))
end

"""
    eoq_quantity(demand_rate, ordering_cost, holding_cost_rate, backlog_cost_rate)

    Computes the economic ordering quantity that minimizes overall costs (ordering costs + holding costs) while meeting  demand.
"""
function eoq_quantity(demand_rate, ordering_cost, holding_cost_rate, backlog_cost_rate)
    return sqrt((2 * demand_rate * ordering_cost) / (holding_cost_rate) * (holding_cost_rate + backlog_cost_rate) / backlog_cost_rate)
end

"""
    eoq_interval(demand_rate, ordering_cost, holding_cost_rate)

    Computes at what interval the economic ordering quantity is ordered.

    See also [`eoq_quantity`](@ref).
"""
function eoq_interval(demand_rate, ordering_cost, holding_cost_rate)
    return sqrt((2 * ordering_cost) / (holding_cost_rate * demand_rate))
end

"""
    eoq_cost_rate(demand_rate, ordering_cost, holding_cost_rate)

    Computes the total cost per time period of ordering the economic ordering quantity.
    
    See also [`eoq_quantity`](@ref).
"""
function eoq_cost_rate(demand_rate, ordering_cost, holding_cost_rate)
    return sqrt(2 * demand_rate * ordering_cost * holding_cost_rate)
end


end # module SupplyChainSimulation
