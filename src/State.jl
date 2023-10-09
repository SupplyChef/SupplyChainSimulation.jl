import Base.push!
import Base.delete!

abstract type InventoryOrderingPolicy end

struct OrderLineTracker
    pending_outbound_order_lines::Dict{L2, Set{OrderLine}} where L2 <: Location
    pending_inbound_order_lines::Dict{L3, Set{OrderLine}} where L3 <: Location
    
    function OrderLineTracker(pending_outbound_order_lines)
        olt = new(Dict{Location, Set{OrderLine}}(), Dict{Location, Set{OrderLine}}())
        for order_line in collect(Base.Iterators.flatten(values(pending_outbound_order_lines)))
            push!(olt, order_line)
        end
        return olt
    end
end

"""
Contains information about the current state of the simulation, including inventory positions and pending orders.
"""
struct State
    on_hand_inventory::Dict{Storage, Dict{Product, Int64}}

    in_transit_inventory::Dict{L1, Dict{P1, Array{Int64, 1}}} where L1 <: Location where P1 <: Product

    order_line_tracker::OrderLineTracker
    filled_order_lines::Set{OrderLine}

    demand::Dict{Tuple{Customer, Product}, Array{Int64, 1}}

    policies::Dict{Tuple{Transport, Product}, InventoryOrderingPolicy}

    historical_on_hand::Array{Dict{Storage, Dict{Product, Int64}}, 1}
    historical_orders::Array{Order, 1}
    historical_transportation::Dict{Trip, Array{OrderLine, 1}}
    historical_filled_order_lines::Array{Set{OrderLine}}
    historical_pending_outbound_order_lines::Array{Dict{Location, Set{OrderLine}}}

    function State(;on_hand_inventory, 
                    in_transit_inventory=Dict{Location, Dict{Product, Array{Int64, 1}}}(), 
                    pending_outbound_order_lines=OrderLine[], 
                    demand, 
                    policies)
        return new(on_hand_inventory, 
                   in_transit_inventory, 
                   OrderLineTracker(pending_outbound_order_lines), 
                   Set{OrderLine}(), 
                   demand, 
                   policies, 
                   [], 
                   Order[],
                   Dict{Trip, Array{OrderLine, 1}}(), 
                   [],
                   [])
    end
end

function push!(olt::OrderLineTracker, order_line::OrderLine)
    if !haskey(olt.pending_outbound_order_lines, order_line.order.origin) 
        olt.pending_outbound_order_lines[order_line.order.origin] = Set{OrderLine}()
    end
    if !haskey(olt.pending_inbound_order_lines, order_line.order.destination) 
        olt.pending_inbound_order_lines[order_line.order.destination] = Set{OrderLine}()
    end

    Base.push!(olt.pending_outbound_order_lines[order_line.order.origin], order_line)
    Base.push!(olt.pending_inbound_order_lines[order_line.order.destination], order_line)
end

function add_order_line!(state::State, order_line::OrderLine)
    olt = state.order_line_tracker
    push!(olt, order_line)
end

function delete_order_line!(state::State, order_line::OrderLine)
    olt = state.order_line_tracker
    Base.delete!(olt.pending_outbound_order_lines[order_line.order.origin], order_line)
    Base.delete!(olt.pending_inbound_order_lines[order_line.order.destination], order_line)
end

function delete_order_lines!(state::State, order_lines::Set{OrderLine})
    for order_line in order_lines
        delete_order_line!(state, order_line)
    end
end

function add_in_transit_inventory!(state::State, to::Location, product::Product, time::Int64, quantity::Int64)
    if !haskey(state.in_transit_inventory, to)
        state.in_transit_inventory[to] = Dict{Product, Array{Float64, 1}}()
    end
    if !haskey(state.in_transit_inventory[to], product)
        state.in_transit_inventory[to][product] = zeros(get_horizon(state))
    end
    state.in_transit_inventory[to][product][time] += quantity
end

function delete_in_transit_inventory!(state::State, to::Location, product::Product, time::Int64, quantity::Int64)
    state.in_transit_inventory[to][product][time] -= quantity
end

function get_in_transit_inventory(state::State, to::Location, product::Product, time::Int64)::Int64
    if !haskey(state.in_transit_inventory, to)
        return 0
    end
    if !haskey(state.in_transit_inventory[to], product)
        return 0
    end
    return state.in_transit_inventory[to][product][time]
end

function get_horizon(state)
    return maximum(length.(values(state.demand)))
end

function snapshot_state!(state::State, time)
    push!(state.historical_on_hand, Dict(k => deepcopy(v) for (k, v) in state.on_hand_inventory))
    push!(state.historical_filled_order_lines, copy(state.filled_order_lines))
    empty!(state.filled_order_lines)
    push!(state.historical_pending_outbound_order_lines, Dict(k => copy(v) for (k, v) in state.order_line_tracker.pending_inbound_order_lines))
    #println("On hand at $time, $(state.on_hand_inventory)")
end

function get_net_inventory(state::State, location::Location, product::Product, time::Int64)
    # on-hand + in-transit + on-order from suppliers - on-order from supplied
    return state.on_hand_inventory[location][product] +
            sum(get_in_transit_inventory(state, location, product, t) for t in time:get_horizon(state)) +
            get_inbound_orders(state, location, product, time) -
            get_outbound_orders(state, location, product, time)
end

function get_inbound_orders(state::State, location::Location, product::Product, time::Int64)::Int64
    sum(ol -> (ol.product == product && ol.order.due_date >= time) ? ol.quantity : 0, 
        get(state.order_line_tracker.pending_inbound_order_lines, location, OrderLine[]);
        init = 0
    )
end

function get_outbound_orders(state::State, location::Location, product::Product, time::Int64)::Int64
    sum(ol -> (ol.product == product && ol.order.due_date >= time) ? ol.quantity : 0, 
        get(state.order_line_tracker.pending_outbound_order_lines, location, OrderLine[]);
        init = 0
    )
end

function get_net_network_inventory(state, location, product)
end

function get_used_trucks(state)
    return [trip.truck for trip in keys(state.historical_transportation)]
end

function get_fixed_transportation_costs(state)
    return sum(t.fixed_cost for t in get_used_trucks(state))
end