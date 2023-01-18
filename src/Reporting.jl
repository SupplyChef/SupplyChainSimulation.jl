function get_total_demand(state)
    demand = 0.0
    for o in filter(o -> isa(o.destination, Customer), state.historical_orders)
        for ol in o.lines
            demand += ol.quantity
        end
    end
    return demand
end

function get_total_sales(state)
    sales = 0
    for ol in filter(ol -> isa(ol.order.destination, Customer), state.historical_lines_filled)
        sales += ol.quantity
    end
    return sales
end

function get_total_lost_sales(state)
    return get_total_demand(state) - get_total_sales(state)
end

function get_total_on_hand(state)
    on_hand = 0
    for historical_on_hand in state.historical_on_hand
        on_hand = reduce(+, values(historical_on_hand), init=0.0)
    end
    return on_hand
end

function get_total_orders(state)
    orders = 0.0
    for o in state.historical_orders
        for ol in o.lines
            orders += ol.quantity
        end
    end
    return orders
end