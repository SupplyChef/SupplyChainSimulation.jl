"""
    get_total_demand(state)

    Gets the total demand (in unit) for the state.
"""
function get_total_demand(state)
    demand = 0.0
    for o in filter(o -> isa(o.destination, Customer), collect(Base.Iterators.flatten(state.historical_orders)))
        for ol in o.lines
            demand += ol.quantity
        end
    end
    return demand
end

"""
    get_total_sales(state)

    Gets the total sales (in unit) for the state.
"""
function get_total_sales(state)
    sales = 0
    for ol in filter(ol -> isa(ol.order.destination, Customer), collect(Base.Iterators.flatten(state.historical_filled_order_lines)))
        sales += ol.quantity
    end
    return sales
end

"""
    get_total_lost_sales(state)

    Gets the total lost sales (in unit) for the state.
"""
function get_total_lost_sales(state)
    return get_total_demand(state) - get_total_sales(state)
end

function get_total_on_hand(state)
    on_hand = 0
    for historical_on_hand in state.historical_on_hand
        on_hand += reduce(+, map(h -> reduce(+, values(h), init=0.0), values(historical_on_hand)), init=0.0)
    end
    return on_hand
end

function get_total_orders(state)
    orders = 0.0
    for o in Base.Iterators.flatten(state.historical_orders)
        for ol in o.lines
            orders += ol.quantity
        end
    end
    return orders
end

"""
    get_total_trip_unit_costs(state)

    Gets the total transportation unit costs.
"""
function get_total_trip_unit_costs(state)
    transportation_costs = 0
    for (trip, order_lines) in state.historical_transportation
        transportation_costs += trip.route.unit_cost * sum(order_line.quantity for order_line in order_lines)
    end
    return transportation_costs
end

"""
    get_total_trip_fixed_costs(state)

    Gets the total transportation fixed costs.
"""
function get_total_trip_fixed_costs(state)
    transportation_costs = 0
    seen = Set{Trip}() 
    for (trip, order_lines) in state.historical_transportation
        if trip ∉ seen
            transportation_costs += get_fixed_cost(trip.route)
        end
        push!(seen, trip)
    end
    return transportation_costs
end

"""
    get_total_holding_costs(state)

    Gets the total holding costs for the state.
"""
function get_total_holding_costs(state)
    holding_costs = 0
    for historical_on_hand in state.historical_on_hand
        for location in keys(historical_on_hand)
            for product in keys(historical_on_hand[location])
                holding_costs += historical_on_hand[location][product] * get(location.holding_costs, product, 0)
            end
        end
    end
    return holding_costs
end
