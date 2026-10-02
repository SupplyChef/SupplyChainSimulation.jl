"""
    recommend_orders(supplychain::SupplyChain, policies; time::Int=1, customer_backlog::Bool=false)

What to order now: the quantities `policies` place on each `(lane, product)` at
period `time` of `supplychain`, as a `Dict{Tuple{Lane, Product}, Int64}` (only
entries with a positive quantity).

This runs the real `simulate` and reads back the orders it placed that period
(`state.historical_orders`, which `simulate` fills by default), so the recommendation comes from exactly the code
path - policy dispatch, minimum-quantity rounding, downstream-to-upstream
ordering - the simulator validates. Nothing is reimplemented.

"Today" is whatever `supplychain` says it is at period 1: each `Storage`'s
`initial_inventory` is the on-hand stock and each `Lane`'s `initial_arrivals`
is the stock in transit, and the demand arrays are the forecast starting at
period 1. Build `supplychain` from current data, with a horizon covering at
least the longest lead time plus the longest forward-coverage window. An
order placed at `time` does not depend on any later period, so a longer
horizon is correct, just slower. Demand is read as integers (`floor`) and
quantities are integers, as in `simulate`.

Customer demand is not an order: only replenishment orders (non-`Customer`
destinations) are returned. An order is returned for the period in which it is
placed, i.e. when a trip departs on that lane; a lane with no departure at
`time` has no entry. `time > 1` simulates forward from the initial state, which
is useful for what-if analysis but does not reflect real stock.

Policies that read past orders (`required_lookback(policy) > 0`, e.g.
`BackwardCoverageOrderingPolicy`) are rejected with an `ArgumentError`: there is
no real order history to feed them, so their recommendation would be wrong.
`supplychain` and `policies` are not modified.
"""
function recommend_orders(supplychain::SupplyChain, policies::Dict{Tuple{Lane, Product}, <:InventoryOrderingPolicy}; time::Int=1, customer_backlog::Bool=false)
    if time < 1 || time > supplychain.horizon
        throw(ArgumentError("time must be between 1 and the supply chain's horizon ($(supplychain.horizon)), got $time"))
    end
    for ((lane, product), policy) in policies
        if required_lookback(policy) > 0
            throw(ArgumentError("recommend_orders cannot be used with $(typeof(policy)) for ($lane, $product): it reads past orders, and no order history is available"))
        end
    end

    final_state = simulate(supplychain, policies; customer_backlog=customer_backlog)

    recommended = Dict{Tuple{Lane, Product}, Int64}()
    for order in Base.Iterators.flatten(final_state.historical_orders)
        (order.creation_time == time && !(order.destination isa Customer) && !ismissing(order.trip)) || continue
        order.quantity > 0 || continue
        key = (order.trip.route, order.product)
        recommended[key] = get(recommended, key, 0) + order.quantity
    end
    return recommended
end
