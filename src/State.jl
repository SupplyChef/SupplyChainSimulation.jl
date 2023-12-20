import Base.push!
import Base.delete!

"""
Contains information about the current state of the simulation, including inventory positions and pending orders.
"""
mutable struct State
    on_hand_inventory::Dict{Tuple{Storage, Product}, Int64}

    in_transit_inventory::Dict{Tuple{<:Node, Product}, Array{Int64, 1}}

    pending_outbound_order_lines::Dict{Tuple{<:Node, Product}, Set{OrderLine}}
    pending_inbound_order_lines::Dict{Tuple{<:Node, Product}, Set{OrderLine}}

    filled_orders::Set{OrderLine}
    placed_orders::Set{OrderLine}

    demand::Dict{Tuple{Customer, Product}, Demand}

    historical_on_hand::Array{Dict{Tuple{Storage, Product}, Int64}, 1}
    historical_orders::Array{Set{OrderLine}, 1}
    historical_transportation::Set{Trip}
    historical_filled_orders::Array{Set{OrderLine}}
    #historical_pending_outbound_order_lines::Array{Dict{Node, Set{OrderLine}}}

    function State(;pending_outbound_order_lines=Dict{Storage, Array{OrderLine, 1}}(), 
                    demand::Array{Demand, 1})
        state = new(Dict{Tuple{Storage, Product}, Int64}(), 
                   Dict{Tuple{<:Node, Product}, Array{Int64, 1}}(), 
                   Dict{Tuple{<:Node, Product}, Set{OrderLine}}(),
                   Dict{Tuple{<:Node, Product}, Set{OrderLine}}(),
                   Set{OrderLine}(),
                   Set{OrderLine}(),
                   Dict{Tuple{Customer, Product}, Demand}((d.customer, d.product) => d for d in demand), 
                   [], 
                   OrderLine[],
                   Set{Trip}(), 
                   [])
                   #,[])
                   
        for order_line in collect(Base.Iterators.flatten(values(pending_outbound_order_lines)))
            add_order_line!(state, order_line)
        end

        return state
    end
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

function delete_order_lines!(state::State, order_lines::Set{OrderLine})
    for order_line in order_lines
        delete_order_line!(state, order_line)
    end
end

function set_on_hand_inventory!(state::State, to::Node, product::Product, quantity)
    state.on_hand_inventory[(to, product)] = Int(quantity)
end

function add_on_hand_inventory!(state::State, to::Node, product::Product, quantity::Int64)
    state.on_hand_inventory[(to, product)] = get(state.on_hand_inventory, (to, product), 0) + quantity
end

function get_on_hand_inventory(state::State, to::Node, product::Product)::Int64
    return get(state.on_hand_inventory, (to, product), 0)
end

function add_in_transit_inventory!(state::State, to::N, product::Product, time::Int64, quantity::Int64) where N <: Node
    if !haskey(state.in_transit_inventory, (to, product))
        state.in_transit_inventory[(to, product)] = zeros(get_horizon(state))
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
    if !haskey(state.in_transit_inventory, (to, product))
        return 0
    end
    return state.in_transit_inventory[(to, product)][time]
end

function get_in_transit_inventories(state::State, to::N, product::Product)::Array{Int64, 1} where N <: Node
    if !haskey(state.in_transit_inventory, (to, product))
        return [0]
    end
    return state.in_transit_inventory[(to, product)]
end

"""
    get_horizon(state::State)

    Gets the number of steps in the simulation.
"""
function get_horizon(state::State)
    return maximum(length.(map(d -> d.demand, values(state.demand))))
end

function snapshot_state!(state::State, time)
    push!(state.historical_on_hand, copy(state.on_hand_inventory))
    push!(state.historical_filled_orders, copy(state.filled_orders))
    empty!(state.filled_orders)
    #state.filled_orders = Set{OrderLine}()
    push!(state.historical_orders, copy(state.placed_orders))
    empty!(state.placed_orders)
    #state.placed_orders = Set{OrderLine}()
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

function get_past_inbound_orders(state::State, location::Node, product::Product, time::Int64, step_back::Int64)::Array{Union{Missing, Int64}, 1}
    past_orders = zeros(Union{Missing, Int64}, step_back)
    for t in 1:step_back
        if (time+1) - t < 1
            past_orders[t] = missing
        else
            order_lines = filter(o -> o.creation_time == time - t && o.product == product && o.origin == location, state.historical_orders[(time+1) - t])
            past_orders[t] = sum(ol -> ol.quantity, order_lines; init=0)
        end
    end
    past_orders
end

function get_net_network_inventory(state, location, product)
end

function get_used_trucks(state)
    return [trip.truck for trip in state.historical_transportation]
end