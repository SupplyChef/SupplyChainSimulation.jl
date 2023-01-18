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