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

    # Ad-valorem tariff cost accrued at fill time (see record_fill!,
    # Simulation.jl) for a tariff-relevant product's shipment - a Supplier-
    # origin lane priced statically, a Storage-origin lane priced against its
    # on-hand cohort's country-of-origin breakdown (see TariffContext,
    # State.jl). Always 0.0 for a supply chain with no Tariffs registered.
    tariff_costs::Float64

    # Purchase/production cost accrued at order placement (see
    # record_purchase!, Simulation.jl): quantity * the order's origin
    # Supplier/Plant `unit_cost` for the product. Accrued when the order is
    # placed, not when it is received or sold, so it also charges stock
    # still on hand or in transit at the horizon's end. Not part of
    # metrics_cost_function (see profit_cost_function).
    purchase_costs::Float64

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

    # Cash flows by period (index = period, 1:horizon), net-cash convention:
    # `cash_out` is what is paid out when it is paid - supplier/plant purchases
    # per the supplier's PaymentTerms (deposit at order, balance relative to
    # shipment), plus freight and tariffs at shipment; `cash_in` is sales
    # receipts, received at shipment to the customer. Holding, overflow and
    # backlog costs are accruals, not cash, and are in neither. A balance
    # payment can be dated before the period being simulated (a negative
    # balance_offset), so both are only final once `simulate` has finished.
    cash_out::Vector{Float64}
    cash_in::Vector{Float64}

    # Cost of financing the net cash position: cost_of_capital times the sum,
    # over periods, of the cumulative net cash out (cumulative cash_out minus
    # cash_in) at the end of the period, floored at 0. This is the money tied up
    # in inventory, in transit and paid ahead of receipt, less what customers
    # have paid. Set at the end of `simulate` (see finalize_cash!); 0.0 when
    # SupplyChain.cost_of_capital is 0. Not part of metrics_cost_function or
    # profit_cost_function (see cash_cost_function).
    capital_costs::Float64

    # The highest cumulative net cash out reached in any period (floored at 0):
    # the cash needed to fund the chain, to compare against a budget. Set at the
    # end of `simulate` (see finalize_cash!).
    peak_cash_outlay::Float64

    # Boolean matrix indexed by [lane_index, departure_time] to track seen trips.
    # Completely avoids Set{Trip} allocation and hashing overhead on the hot path.
    seen_trips::Matrix{Bool}

    SimMetrics() = new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Float64[], Float64[], 0.0, 0.0, Matrix{Bool}(undef, 0, 0))
    SimMetrics(num_lanes::Int, horizon::Int) = new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zeros(horizon), zeros(horizon), 0.0, 0.0, zeros(Bool, num_lanes, horizon))
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
    metrics.tariff_costs = 0.0
    metrics.purchase_costs = 0.0
    metrics.backlog = 0.0
    fill!(metrics.cash_out, 0.0)
    fill!(metrics.cash_in, 0.0)
    metrics.capital_costs = 0.0
    metrics.peak_cash_outlay = 0.0
    fill!(metrics.seen_trips, false)
    return metrics
end

# Adds `out`/`in` to the cash flows of `period`. A period outside the horizon
# (e.g. a balance due after it) is not part of the curve and is dropped.
@inline function _add_cash!(metrics::SimMetrics, period::Int, cash_out::Float64, cash_in::Float64=0.0)
    if 1 <= period <= length(metrics.cash_out)
        @inbounds metrics.cash_out[period] += cash_out
        @inbounds metrics.cash_in[period] += cash_in
    end
    return nothing
end

"""
    get_cumulative_net_cash_out(metrics::SimMetrics)::Vector{Float64}

The cumulative net cash out at the end of each period: everything paid out so far minus everything
received. Negative once customers have paid more than the chain has spent.
"""
get_cumulative_net_cash_out(metrics::SimMetrics) = cumsum(metrics.cash_out .- metrics.cash_in)

"""
    finalize_cash!(metrics::SimMetrics, cost_of_capital::Float64)

Derives `peak_cash_outlay` and `capital_costs` from the cash flows once they are final, i.e. at the end
of `simulate`.
"""
function finalize_cash!(metrics::SimMetrics, cost_of_capital::Float64)
    cumulative = 0.0
    peak = 0.0
    employed = 0.0
    for t in eachindex(metrics.cash_out)
        cumulative += metrics.cash_out[t] - metrics.cash_in[t]
        peak = max(peak, cumulative)
        employed += max(cumulative, 0.0)
    end
    metrics.peak_cash_outlay = peak
    metrics.capital_costs = cost_of_capital * employed
    return metrics
end
