"""
Orders up to a given number based on the number of units on hand; no matter what is on order.
"""
mutable struct OnHandUptoOrderingPolicy <: InventoryOrderingPolicy
    upto
end

function get_parameter_count(policy::OnHandUptoOrderingPolicy)
    return 1
end

function set_parameter!(policy::OnHandUptoOrderingPolicy, values)
    policy.upto = Int(round(values[1]))
end

function get_order(policy::OnHandUptoOrderingPolicy, state, network, location, lane, product, time)
    return max(0, policy.upto - state.on_hand_inventory[location][product])
end

"""
Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog).
"""
mutable struct NetUptoOrderingPolicy <: InventoryOrderingPolicy
    upto
end

function get_parameter_count(policy::NetUptoOrderingPolicy)
    return 1
end

function set_parameter!(policy::NetUptoOrderingPolicy, values)
    policy.upto = Int(round(values[1]))
end

function get_order(policy::NetUptoOrderingPolicy, state, network, location, lane, product, time)
    return max(0, policy.upto - get_net_inventory(state, location, product, time))
end

"""
Orders the same amount; no matter the scenario.
"""
mutable struct FixedOrderingPolicy <: InventoryOrderingPolicy
    orders
end

function get_parameter_count(policy::FixedOrderingPolicy)
    return length(policy.orders)
end

function set_parameter!(policy::FixedOrderingPolicy, values)
    policy.orders = values
end

function get_order(policy::FixedOrderingPolicy, state, network, lane, product, time)
    return policy.orders[time]
end

"""
Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog) if the net inventory is below a threshold.
"""
mutable struct NetSSOrderingPolicy <: InventoryOrderingPolicy
    s
    S
end

function get_parameter_count(policy::NetSSOrderingPolicy)
    return 2
end

function set_parameter!(policy::NetSSOrderingPolicy, values)
    policy.s = Int(round(values[1]))
    policy.S = Int(round(values[2]))
end

function get_order(policy::NetSSOrderingPolicy, state, network, location, lane, product, time)
    net_inventory = get_net_inventory(state, location, product, time)
    if net_inventory >= policy.s
        return 0
    else
        return max(0, policy.S - net_inventory)
    end
end