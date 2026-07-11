"""
    required_lookback(policy::InventoryOrderingPolicy)::Int

The number of past periods' outbound order quantities (see
`get_past_outbound_orders`) `policy` needs to make its ordering decisions,
or `0` (the default, for every policy except `BackwardCoverageOrderingPolicy`)
if it doesn't look at history at all.

`Env` uses this at construction time to decide whether `State` needs to
maintain `outbound_order_quantities` for the run: recording it is skipped
entirely - no allocation, no per-order bookkeeping - unless some policy in
play actually declares a nonzero requirement here. Override this for any
future policy that needs to look backward the way
`BackwardCoverageOrderingPolicy` does.
"""
required_lookback(policy::InventoryOrderingPolicy)::Int = 0

"""
Orders a given quantity specific to each time period.
"""
mutable struct QuantityOrderingPolicy <: InventoryOrderingPolicy
    orders::Array{Int64, 1}
end

"""
    get_parameters(policy::QuantityOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::QuantityOrderingPolicy)
    return policy.orders
end

function set_parameters!(policy::QuantityOrderingPolicy, values::Array{Float64, 1})
    policy.orders .= Int.(round.(values))
end

function get_order(policy::QuantityOrderingPolicy, state::State, env::Env, location, lane, product, time)::Int64
    order = max(0, policy.orders[time])
    return order
end

"""
Orders a given quantity at a given time period.
"""
mutable struct ProductQuantityOrderingPolicy <: InventoryOrderingPolicy
    order::Int64
    period::Int64
end

"""
    get_parameters(policy::ProductQuantityOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::ProductQuantityOrderingPolicy)
    return [policy.order]
end

function set_parameters!(policy::ProductQuantityOrderingPolicy, values::Array{Float64, 1})
    policy.order = Int(round(values[1]))
end

function get_order(policy::ProductQuantityOrderingPolicy, state::State, env::Env, location, lane, product, time)::Int64
    if time == policy.period
        order = max(0, policy.order)
        return order
    else
        return 0
    end 
end

"""
Orders up to a given number based on the number of units on hand; no matter what is on order.
"""
mutable struct OnHandUptoOrderingPolicy <: InventoryOrderingPolicy
    upto::Int64
end


"""
    get_parameters(policy::OnHandUptoOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::OnHandUptoOrderingPolicy)
    return [policy.upto]
end

function set_parameters!(policy::OnHandUptoOrderingPolicy, values::Array{Float64, 1})
    policy.upto = Int(round(values[1]))
end

function get_order(policy::OnHandUptoOrderingPolicy, state::State, env::Env, location, lane, product, time)::Int64
    return max(0, policy.upto - get_on_hand_inventory(state, location, product))
end

"""
Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog).
"""
mutable struct NetUptoOrderingPolicy <: InventoryOrderingPolicy
    upto::Int64
end


"""
    get_parameters(policy::NetUptoOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::NetUptoOrderingPolicy)
    return [policy.upto]
end

function set_parameters!(policy::NetUptoOrderingPolicy, values::Array{Float64, 1})
    policy.upto = Int(round(values[1]))
end

function get_order(policy::NetUptoOrderingPolicy, state::State, env::Env, location, lane, product, time)::Int64
    return max(0, policy.upto - get_net_inventory(state, location, product, time))
end

"""
Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog) if the net inventory is below a threshold.
"""
mutable struct NetSSOrderingPolicy <: InventoryOrderingPolicy
    s::Int64
    S::Int64
end

"""
    get_parameters(policy::NetSSOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::NetSSOrderingPolicy)
    return [policy.s, policy.S]
end

function set_parameters!(policy::NetSSOrderingPolicy, values::Array{Float64, 1})
    policy.s = Int(round(values[1]))
    policy.S = Int(round(values[2]))
end

function get_order(policy::NetSSOrderingPolicy, state::State, env::Env, location, lane, product, time)::Int64
    net_inventory = get_net_inventory(state, location, product, time)
    #println("net inventory @ $time: $net_inventory")
    if net_inventory >= policy.s
        return 0
    else
        return max(0, policy.S - net_inventory)
    end
end

"""
Orders inventory to cover the coming periods based on the mean forecasted demand.
"""
mutable struct ForwardCoverageOrderingPolicy <: InventoryOrderingPolicy
    cover::Float64
end

"""
    get_parameters(policy::ForwardCoverageOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::ForwardCoverageOrderingPolicy)
    return [policy.cover]
end

function set_parameters!(policy::ForwardCoverageOrderingPolicy, values::Array{Float64, 1})
    policy.cover = values[1]
end

function get_order(policy::ForwardCoverageOrderingPolicy, state::State, env::Env, location, lane, product, time)::Int64
    net_inventory = get_net_inventory(state, location, product, time)
    mean_demand = get_mean_demand(env, location, product, time)

    coverage = 0
    cover = policy.cover
    t = time
    while cover > 0
        if cover >= 1
            coverage = coverage + mean_demand[min(t, end)]
            cover = cover - 1
            t = t + 1
        else
            coverage = coverage + cover * mean_demand[min(t, end)]
            cover = 0
        end
    end
    # coverage - net_inventory can come out NaN/Inf for degenerate optimizer
    # candidates (e.g. mean_demand summing to values that make later float
    # arithmetic non-finite); guard rather than let Int(ceil(...)) throw.
    deficit = coverage - net_inventory
    if !isfinite(deficit)
        @warn "ForwardCoverageOrderingPolicy.get_order: non-finite deficit, falling back to order=0" cover=policy.cover coverage net_inventory maxlog=50
    end
    order = isfinite(deficit) ? max(0, ceil(Int, deficit)) : 0
    #println("cover $(policy.cover); mean demand $mean_demand; coverage $coverage; net inventory $net_inventory; order $order; time $time")
    return order
end

"""
Orders inventory to cover the coming periods based on past demand.
"""
mutable struct BackwardCoverageOrderingPolicy <: InventoryOrderingPolicy
    cover::Array{Float64, 1}
end

"""
    get_parameters(policy::BackwardCoverageOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::BackwardCoverageOrderingPolicy)
    return policy.cover
end

function set_parameters!(policy::BackwardCoverageOrderingPolicy, values::Array{Float64, 1})
    policy.cover = values
end

"""
    required_lookback(policy::BackwardCoverageOrderingPolicy)::Int

`get_order` below looks back `length(policy.cover)` periods via
`get_past_outbound_orders`, so that's exactly how far `Env` needs to keep
`outbound_order_quantities` for this policy's `(location, product)`.
"""
required_lookback(policy::BackwardCoverageOrderingPolicy)::Int = length(policy.cover)

function get_order(policy::BackwardCoverageOrderingPolicy, state::State, env::Env, location, lane, product, time)::Int64
    net_inventory = get_net_inventory(state, location, product, time)
    
    past_orders = get_past_outbound_orders(state, location, product, time, length(policy.cover))
    #println("$lane $time $location $past_orders")
    
    weights = 0
    coverage = 0
    for i in 1:length(policy.cover) - 1
        if !ismissing(past_orders[i])
            coverage += policy.cover[i] * past_orders[i]
            weights += policy.cover[i]
        end
    end

    if weights != 0
        coverage = coverage / (weights / sum(policy.cover))
    end

    coverage = coverage + policy.cover[end]

    # coverage - net_inventory can come out NaN/Inf for degenerate optimizer
    # candidates (e.g. weights/sum(policy.cover) dividing by ~0); guard
    # rather than let Int(ceil(...)) throw.
    deficit = coverage - net_inventory
    if !isfinite(deficit)
        @warn "BackwardCoverageOrderingPolicy.get_order: non-finite deficit, falling back to order=0" cover=policy.cover coverage weights net_inventory maxlog=50
    end
    order = isfinite(deficit) ? max(0, ceil(Int, deficit)) : 0
    @debug "Computing order at $time, $location, $product, order: $order, past outbound orders: $past_orders, cover: $coverage, net inventory: $net_inventory"
    return order
end

"""
Places a single order at a given time.
"""
mutable struct SingleOrderOrderingPolicy <: InventoryOrderingPolicy
    period::Int64
    quantity
end

function get_parameters(policy::SingleOrderOrderingPolicy)
    return [policy.period, policy.quantity]
end

function set_parameters!(policy::SingleOrderOrderingPolicy, values::Array{Float64, 1})
    policy.quantity = Int(round(values[2]))
end

function get_order(policy::SingleOrderOrderingPolicy, state::State, env::Env, location, lane, product, time)::Int64
    if time == policy.period
        return policy.quantity
    else
        return 0
    end
end

function get_order(policy::InventoryOrderingPolicy, state::State, env::Env, location, lane, product, time)::Int64
    return 0
end