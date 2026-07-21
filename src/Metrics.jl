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

    # Boolean matrix indexed by [lane_index, departure_time] to track seen trips.
    # Completely avoids Set{Trip} allocation and hashing overhead on the hot path.
    seen_trips::Matrix{Bool}

    SimMetrics() = new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Matrix{Bool}(undef, 0, 0))
    SimMetrics(num_lanes::Int, horizon::Int) = new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zeros(Bool, num_lanes, horizon))
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
    fill!(metrics.seen_trips, false)
    return metrics
end
