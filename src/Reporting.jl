"""
    get_total_demand(state)

    Gets the total demand (in unit) for the state.
"""
function get_total_demand(state)
    demand = 0.0
    for ol in filter(ol -> isa(ol.destination, Customer), collect(Base.Iterators.flatten(state.historical_orders)))
        demand += ol.quantity * state.demand[(ol.destination, ol.product)].sales_price
    end
    return demand
end

"""
    get_total_sales(state)

    Gets the total sales (in unit) for the state.
"""
function get_total_sales(state)
    sales = 0.0
    for ol in filter(ol -> isa(ol.destination, Customer), collect(Base.Iterators.flatten(state.historical_filled_orders)))
        sales += ol.quantity * state.demand[(ol.destination, ol.product)].sales_price
    end
    return sales
end

"""
    get_total_lost_sales(state)

    Gets the total lost sales (in unit) for the state.
"""
function get_total_lost_sales(state)
    all_orders = Set(filter(ol -> isa(ol.destination, Customer), collect(Base.Iterators.flatten(state.historical_orders))))
    fulfilled_orders = Set(filter(ol -> isa(ol.destination, Customer), collect(Base.Iterators.flatten(state.historical_filled_orders))))
    unfulfilled_orders = setdiff(all_orders, fulfilled_orders)

    lost_sales = 0.0
    for ol in unfulfilled_orders
        lost_sales += ol.quantity * state.demand[(ol.destination, ol.product)].lost_sales_cost
    end
    return lost_sales
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
        orders += o.quantity
    end
    return orders
end

"""
    get_total_trip_unit_costs(state)

    Gets the total transportation unit costs.
"""
function get_total_trip_unit_costs(state)
    transportation_costs = 0.0
    for filled_orders in state.historical_filled_orders
        transportation_costs += sum(order_line.trip.route.unit_cost * order_line.quantity for order_line in filled_orders; init=0.0)
    end
    return transportation_costs
end

"""
    get_total_trip_fixed_costs(state)

    Gets the total transportation fixed costs.
"""
function get_total_trip_fixed_costs(state)
    transportation_costs = 0
    for trip in state.historical_transportation
        transportation_costs += get_fixed_cost(trip.route)
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
        for (location, product) in keys(historical_on_hand)
            holding_costs += historical_on_hand[(location, product)] * get(location.unit_holding_cost, product, 0)
        end
    end
    return holding_costs
end

function get_shipments(state)
    state.historical_filled_orders
end