abstract type InventoryOrderingPolicy end

struct State
    on_hand_inventory::Dict{Storage, Dict{Product, Int64}}

    in_transit_inventory::Dict{L1, Dict{P1, Array{Int64, 1}}} where L1 <: Location where P1 <: Product

    pending_order_lines::Dict{L2, Array{OrderLine, 1}} where L2 <: Location

    demand::Dict{Tuple{Customer, Product}, Array{Int64, 1}}

    policies::Dict{Tuple{Transport, Product}, InventoryOrderingPolicy}

    historical_on_hand::Array{Dict{Storage, Dict{Product, Int64}}, 1}
    historical_orders::Array{Order, 1}
    historical_transportation::Dict{Trip, Array{OrderLine, 1}}
    historical_lines_filled::Array{OrderLine, 1}

    function State(;on_hand_inventory, in_transit_inventory, pending_order_lines, demand, policies)
        return new(on_hand_inventory, in_transit_inventory, pending_order_lines, demand, policies, [], Order[], Dict{Trip, Array{OrderLine, 1}}(), OrderLine[])
    end
end

function snapshot_state!(state, time)
    push!(state.historical_on_hand, copy(state.on_hand_inventory))
    #println("On hand at $time, $(state.on_hand_inventory)")
end

function get_net_inventory(state, location, product, time)
    # on-hand + in-transit + on-order from suppliers - on-order from supplied
    return state.on_hand_inventory[location][product] +
            sum(state.in_transit_inventory[location][product][time:end]) +
            get_inbound_orders(state, location, product, time) -
            get_outbound_orders(state, location, product, time)
end

function get_inbound_orders(state, location, product, time)
    reduce(+,
        map(ol -> ol.quantity, 
            filter(ol -> location ∈ get_destinations(ol.order.trip.route) && 
                         ol.product == product && 
                         ol.order.due_date >= time, 
                         vcat([state.pending_order_lines[l] for l in keys(state.pending_order_lines)]...)
            )
        ),
        init = 0.0
    )
end

function get_outbound_orders(state, location, product, time)
    reduce(+,
        map(ol -> ol.quantity, 
            filter(ol -> ol.product == product && 
                         ol.order.due_date >= time, 
                         state.pending_order_lines[location]
            )
        ),
        init = 0.0
    )
end

function get_net_network_inventory(state, location, product)
end

function get_used_trucks(state)
    return [trip.truck for trip in keys(state.historical_transportation)]
end

function get_transportation_costs(state)
    return sum(t.fixed_cost for t in get_used_trucks(state))
end

function get_holding_costs(state)
    holding_costs = 0
    for historical_on_hand in state.historical_on_hand
        for location in keys(historical_on_hand)
            for product in keys(historical_on_hand[location])
                holding_costs += historical_on_hand[location][product] * product.holding_costs
            end
        end
    end
    return holding_costs
end
