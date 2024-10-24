module SupplyChainSimulation

export Network
export Product
export Bundle
export Order
export OrderLine

export State

export InventoryOrderingPolicy
export OnHandUptoOrderingPolicy
export NetUptoOrderingPolicy
export NetSSOrderingPolicy
export ForwardCoverageOrderingPolicy
export BackwardCoverageOrderingPolicy
export QuantityOrderingPolicy

export set_parameters!
export get_downstream_customers
export simulate
export optimize!

export get_sorted_locations
export get_inbound_orders
export get_outbound_orders
export get_net_inventory
export get_holding_costs

export get_total_orders
export get_total_trip_unit_costs
export get_total_trip_fixed_costs
export get_total_demand
export get_total_sales
export get_total_lost_sales
export get_total_holding_costs
export get_on_hand_inventory
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

abstract type InventoryOrderingPolicy end

include("Model.jl")
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
