module SupplyChainSimulation

export Network
export Product
export Bundle
export Order
export OrderLine

export State
export reset!

export Env

export SimMetrics
export get_metrics

export InventoryOrderingPolicy
export OnHandUptoOrderingPolicy
export NetUptoOrderingPolicy
export NetSSOrderingPolicy
export ForwardCoverageOrderingPolicy
export BackwardCoverageOrderingPolicy
export QuantityOrderingPolicy
export required_lookback

export set_parameters!
export get_downstream_customers
export simulate
export optimize!
export metrics_cost_function

export get_sorted_locations
export get_inbound_orders
export get_outbound_orders
export get_past_outbound_orders
export get_net_inventory
export get_holding_costs

export get_total_orders
export get_total_trip_unit_costs
export get_total_trip_fixed_costs
export get_total_demand
export get_total_sales
export get_total_lost_sales
export get_total_holding_costs
export get_total_overflow_costs
export get_on_hand_inventory
export get_overflow_inventory
export remove_on_hand_inventory!

export get_trips

export eoq_quantity

export plot_inventory_movement
export plot_inventory_onhand
# export plot_pending_outbound_order_lines

using Graphs
#using Optim
#using BlackBoxOptim
using SupplyChainModeling
using FunctionWrappers: FunctionWrapper

abstract type InventoryOrderingPolicy end

# Resolves which concrete get_order method to call once, at Env-construction
# time (via wrap_get_order, see Policy.jl), instead of on every call in
# simulate()'s hot loop - policy::InventoryOrderingPolicy is abstract with 8
# concrete subtypes, so calling get_order(policy, ...) directly forces a
# fully dynamic dispatch that boxes every argument at the call site
# regardless of how precisely those arguments are otherwise typed (confirmed
# via code_warntype/Profile.Allocs on the beer game benchmark - this was the
# actual dominant allocator, not the Node-typed arguments the ConcreteNode
# retyping targeted). FunctionWrapper type-erases the resolved closure into
# one fixed concrete struct, so storing/reading it from Trip.policies avoids
# that boxing regardless of how many policy types exist (unlike relying on
# Julia's union-splitting, which only helps for small closed Unions).
#
# The state/env argument positions are Any rather than State/Env: Trip
# (Model-Transportation.jl) needs this alias for its policies field, but
# Trip is needed by Env's own fields (departures, etc.), so State/Env can't
# be referenced concretely here without a circular type dependency. This
# costs nothing in practice - state/env are passed as existing heap
# references (State is a mutable struct, Env is a non-isbits immutable
# struct, so both are already heap-allocated before this call), not boxed
# fresh on every call the way a fully dynamic dispatch's arguments are.
const GetOrderFn = FunctionWrapper{Int64, Tuple{Any, Any, ConcreteNode, Lane, Product, Int64}}

include("Model.jl")
include("Metrics.jl")
include("State.jl")
include("Env.jl")
include("Policy.jl")

include("Optimization.jl")
include("Reporting.jl")
include("Simulation.jl")
include("Visualization.jl")

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
