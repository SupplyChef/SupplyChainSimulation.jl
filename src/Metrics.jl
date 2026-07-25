"""
Running (incrementally-updated) cost and quantity totals for a simulation run.

Unlike the `get_total_*` functions in `Reporting.jl` - which recompute these
totals by scanning `State`'s `historical_*` arrays after the fact -
`SimMetrics` is updated inline, at the point a fill, drop, or order
placement actually happens during `simulate`. Reading a total back out is
then O(1) instead of an O(horizon) (or worse) scan.

Critically, accumulation into `SimMetrics` does not depend on
`Env.record_history`: it always happens, so `optimize!` can score candidate
policies from `state.metrics` alone without needing the per-period history
arrays that only exist to support detailed reporting/visualization after a
run. See `Env.record_history` for what that flag does and does not disable.
"""
mutable struct SimMetrics
    sales::Float64
    lost_sales::Float64
    holding_costs::Float64
    overflow_costs::Float64
    trip_unit_costs::Float64
    trip_fixed_costs::Float64
    orders::Float64
    demand::Float64

    # Sum, across every period closed out so far, of every location's
    # currently-outstanding order-line quantity (see snapshot_state!, which
    # charges this the same way it charges holding_costs). Raw units, not
    # pre-priced - unlike holding_costs (which bakes in each location's own
    # unit_holding_cost) there's no per-location backlog-cost field on
    # Storage/Supplier, so a cost_function applies whatever backlog weight
    # fits its own convention, the same way metrics_cost_function applies its
    # own 0.001 weight to `orders` below.
    # Customer-destined order lines are excluded unless Env.customer_backlog
    # is true: with the default false, a customer order that can't be filled
    # the same period it's created is dropped as a lost sale next period (see
    # record_drop!/place_orders(..., ::Customer, ...)), never a genuine
    # backlog, so counting it here even for the one snapshot before that drop
    # fires would double up with lost_sales for no reason. When
    # customer_backlog is true, customer orders queue like any other node's
    # and are counted the same way (see snapshot_state!).
    backlog::Float64

    # Trips that have already had their one-time fixed cost charged this run.
    # The same Trip (route + departure) can be assigned to many order lines,
    # even across periods, but get_total_trip_fixed_costs only charges it
    # once per run (by scanning the deduplicating Set historical_transportation)
    # - this mirrors that dedup without depending on historical_transportation.
    seen_trips::Set{Trip}

    SimMetrics() = new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Set{Trip}())
end

function reset!(metrics::SimMetrics)
    metrics.sales = 0.0
    metrics.lost_sales = 0.0
    metrics.holding_costs = 0.0
    metrics.overflow_costs = 0.0
    metrics.trip_unit_costs = 0.0
    metrics.trip_fixed_costs = 0.0
    metrics.orders = 0.0
    metrics.demand = 0.0
    metrics.backlog = 0.0
    empty!(metrics.seen_trips)
    return metrics
end
