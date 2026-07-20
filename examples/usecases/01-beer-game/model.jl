#=
Beer Game use case: naive (locally-reactive, untuned) ordering policies vs.
policies jointly tuned by optimize!() across all three upstream echelons,
plus an out-of-sample ("generalization") check of the tuned policy on a
fresh batch of demand scenarios it never saw during tuning.

The "optimized, in-sample" configuration below is deliberately identical to
the repo's own test/policy-beergame-tests.jl beer_game() (same node/lane
setup, same seed, same scenario_count, same policy family) specifically so
its output can be cross-checked against that test's known-good asserted
values (103 lost sales / 1828 sales / 1931 demand for final_states[1]) as a
correctness check on this script, not just on the package.

Run from the repo root after examples/usecases/setup.jl:
    julia --project=examples/usecases examples/usecases/01-beer-game/model.jl
Writes results.json next to this file.
=#
using SupplyChainModeling, SupplyChainSimulation, Random, JSON3
using Distributions: Poisson # `using Distributions` would also import Distributions.Product,
                              # which collides with SupplyChainModeling/SupplyChainSimulation's Product
using Statistics: var, mean, std

const HORIZON = 200
const SCENARIO_COUNT = 30
const CALIBRATION_SEED = 3
const HOLDOUT_SEED = 999
const MEAN_DEMAND = 10.0

product = Product("product")
customer = Customer("customer")
retailer = Storage("retailer");   add_product!(retailer, product;   unit_holding_cost=0.1, initial_inventory=20)
wholesaler = Storage("wholesaler"); add_product!(wholesaler, product; unit_holding_cost=0.1, initial_inventory=20)
factory = Storage("factory");     add_product!(factory, product;     unit_holding_cost=0.1, initial_inventory=20)
supplier = Supplier("supplier")

l  = Lane(retailer, customer;   unit_cost=0)
l2 = Lane(wholesaler, retailer; unit_cost=0, time=2)
l3 = Lane(factory, wholesaler;  unit_cost=0, time=2)
l4 = Lane(supplier, factory;    unit_cost=0, time=4)

function build_scenario()
    network = SupplyChain(HORIZON)
    add_supplier!(network, supplier)
    add_storage!(network, retailer); add_storage!(network, wholesaler); add_storage!(network, factory)
    add_customer!(network, customer)
    add_product!(network, product)
    add_lane!(network, l); add_lane!(network, l2); add_lane!(network, l3); add_lane!(network, l4)
    add_demand!(network, customer, product, rand(Poisson(MEAN_DEMAND), HORIZON) * 1.0; sales_price=1.0, lost_sales_cost=1.0)
    return network
end

function build_scenarios(n, seed)
    Random.seed!(seed)
    return [build_scenario() for _ in 1:n]
end

function aggregate(states)
    total_demand = sum(get_total_demand(s) for s in states)
    total_sales = sum(get_total_sales(s) for s in states)
    total_lost_sales = sum(get_total_lost_sales(s) for s in states)
    return (
        total_demand = total_demand,
        total_sales = total_sales,
        total_lost_sales = total_lost_sales,
        fill_rate = total_sales / total_demand,
    )
end

# --- Bullwhip instrumentation: WHICH echelon amplifies vs. absorbs demand
#     variance, not just the aggregate fill rate. ---

# Per-period quantity ordered on the (origin, destination) lane - i.e. the
# order stream *placed by* `destination` (an OrderLine's `destination` is
# the node receiving/ordering, `origin` is who it ordered from - see
# Model.jl). historical_orders[1] is the pre-simulation (period 0)
# snapshot, so period t's orders live at historical_orders[t+1].
function lane_order_series(state, origin, destination, product, horizon)
    qty = zeros(Float64, horizon)
    for period_orders in state.historical_orders
        for ol in period_orders
            if ol.origin == origin && ol.destination == destination && ol.product == product && 1 <= ol.creation_time <= horizon
                qty[ol.creation_time] += ol.quantity
            end
        end
    end
    return qty
end

# Same period offset as above: historical_on_hand[1] is period 0.
function onhand_series(state, storage, product, horizon)
    qty = zeros(Float64, horizon)
    for (idx, snapshot) in enumerate(state.historical_on_hand)
        t = idx - 1
        if 1 <= t <= horizon
            qty[t] = get(snapshot, (storage, product), 0.0)
        end
    end
    return qty
end

demand_series(state, customer, product) = state.demand[(customer, product)].demand

# bullwhip ratio for a lane = Var(orders placed on that lane) / Var(end
# customer demand) for the same scenario - the standard definition from the
# bullwhip literature (Lee/Padmanabhan/Whang). >1 means this echelon is
# amplifying variance relative to what the end customer actually did; <1
# means it's damping/absorbing it. Averaged (mean of per-scenario ratios,
# not pooled) across all calibration scenarios.
function bullwhip_ratios(states, lanes_named, customer, product, horizon)
    ratios = Dict(name => Float64[] for (name, _, _) in lanes_named)
    for s in states
        d = demand_series(s, customer, product)
        dvar = var(d)
        dvar == 0 && continue
        for (name, origin, destination) in lanes_named
            o = lane_order_series(s, origin, destination, product, horizon)
            push!(ratios[name], var(o) / dvar)
        end
    end
    return Dict(name => mean(v) for (name, v) in ratios)
end

# Coefficient of variation (std/mean) of each node's own on-hand inventory -
# the complementary signal: a node that isn't passing variance through as
# orders (low bullwhip ratio above) has to be *carrying* that variance
# somewhere, and it shows up here as a larger, more variable buffer.
function inventory_cv(states, storages_named, product, horizon)
    cvs = Dict(name => Float64[] for (name, _) in storages_named)
    for s in states
        for (name, storage) in storages_named
            series = onhand_series(s, storage, product, horizon)
            m = mean(series)
            m == 0 && continue
            push!(cvs[name], std(series) / m)
        end
    end
    return Dict(name => mean(v) for (name, v) in cvs)
end

const LANES_NAMED = [("retailer_orders (l2)", wholesaler, retailer), ("wholesaler_orders (l3)", factory, wholesaler), ("factory_orders (l4)", supplier, factory)]
const STORAGES_NAMED = [("retailer", retailer), ("wholesaler", wholesaler), ("factory", factory)]
const BACKLOG_NODES_NAMED = [("wholesaler", wholesaler), ("factory", factory), ("supplier", supplier)]

# --- Cost breakdown: what optimize! actually minimizes, and whether the
#     "optimized" policy really is cheaper than the naive one under that
#     same objective (metrics_cost_function isn't just applied post hoc here
#     - it's the literal function optimize! searched against). ---
function cost_breakdown(states)
    return (
        total_cost = sum(metrics_cost_function(s) for s in states),
        holding_costs = sum(s.metrics.holding_costs for s in states),
        trip_fixed_costs = sum(s.metrics.trip_fixed_costs for s in states),
        trip_unit_costs = sum(s.metrics.trip_unit_costs for s in states),
        order_count = sum(s.metrics.orders for s in states),
        sales = sum(s.metrics.sales for s in states),
        lost_sales = sum(s.metrics.lost_sales for s in states),
    )
end

# --- Backlog by layer: unfulfilled internal replenishment orders queued at
#     each origin node over time. Customer orders never backlog (due_date ==
#     creation_time - unfilled the same period becomes a lost sale, not a
#     queued backorder, per place_orders/send_inventory! in Simulation.jl).
#     Internal replenishment orders (due_date == typemax(Int64)) never
#     expire, so whatever an origin can't ship immediately queues up as a
#     real backorder - reconstructed here as cumulative-ordered minus
#     cumulative-filled at that origin, since neither historical_orders nor
#     historical_filled_orders is a live queue on its own. Both arrays share
#     the same period-offset convention as lane_order_series/onhand_series
#     above (index 1 = period 0).
function backlog_series(state, origin, product, horizon)
    ordered = zeros(Float64, horizon)
    filled = zeros(Float64, horizon)
    for period_orders in state.historical_orders
        for ol in period_orders
            if ol.origin == origin && ol.product == product && 1 <= ol.creation_time <= horizon
                ordered[ol.creation_time] += ol.quantity
            end
        end
    end
    for (idx, period_filled) in enumerate(state.historical_filled_orders)
        t = idx - 1
        if 1 <= t <= horizon
            for ol in period_filled
                if ol.origin == origin && ol.product == product
                    filled[t] += ol.quantity
                end
            end
        end
    end
    return cumsum(ordered) .- cumsum(filled)
end

function backlog_summary(states, nodes_named, product, horizon)
    peak = Dict(name => Float64[] for (name, _) in nodes_named)
    ending = Dict(name => Float64[] for (name, _) in nodes_named)
    for s in states
        for (name, node) in nodes_named
            series = backlog_series(s, node, product, horizon)
            push!(peak[name], maximum(series))
            push!(ending[name], series[end])
        end
    end
    return (
        peak = Dict(name => mean(v) for (name, v) in peak),
        ending = Dict(name => mean(v) for (name, v) in ending),
    )
end

# --- "Classic beer-game" scoring: the standard board-game convention is a
#     holding cost + backlog cost at *every* stage, accumulated over the
#     whole game - not the lost-sales-at-the-customer-interface convention
#     metrics_cost_function/cost_breakdown above actually use. This gets as
#     close to that classic convention as SupplyChainSimulation.jl's model
#     structurally allows: real backlog cost at the three internal echelons
#     (wholesaler/factory/supplier, via backlog_series above), and a
#     lost-sales proxy at the retailer. That proxy is necessary, not a
#     choice: Customer orders in this package always have due_date ==
#     creation_time (see place_orders in Simulation.jl) - unlike every other
#     node type, an unfulfilled customer order is dropped the same period,
#     never queued. In the original board game a stockout at retail is also
#     just a backorder, exactly like every other stage - not a permanently
#     lost sale - so the retailer term here is the closest available proxy,
#     not a true equivalent, and that's a real limitation of this package's
#     Customer node type, not an arbitrary modeling choice made for this post.
#     backlog_rate defaults to 2x holding_rate, the standard ratio in
#     Sterman's canonical version of the game (there: $0.50/case/week
#     holding, $1.00/case/week backlog) - applied here to this network's own
#     0.1 holding rate rather than importing Sterman's absolute numbers,
#     since lead times/initial inventory/demand distribution already differ
#     from his canonical setup.
function classic_score(states, product, horizon; holding_rate=0.1, backlog_rate=0.2)
    per_stage = Dict(name => Float64[] for name in ("retailer", "wholesaler", "factory", "supplier"))
    for s in states
        retailer_holding = holding_rate * sum(onhand_series(s, retailer, product, horizon))
        retailer_shortfall = get_total_lost_sales(s) # proxy - see docstring above
        push!(per_stage["retailer"], retailer_holding + retailer_shortfall)

        for (name, node) in (("wholesaler", wholesaler), ("factory", factory), ("supplier", supplier))
            holding = holding_rate * sum(onhand_series(s, node, product, horizon))
            backlog = backlog_rate * sum(backlog_series(s, node, product, horizon))
            push!(per_stage[name], holding + backlog)
        end
    end
    stage_totals = Dict(name => sum(v) for (name, v) in per_stage)
    return (per_stage = stage_totals, total = sum(values(stage_totals)))
end

# --- Sterman's (1989) "anchor and adjust" heuristic - a published, measured
#     model of how real humans play the beer game, not a policy this package
#     ships. Each period: order = forecast + alpha_stock*(desired_stock -
#     on_hand) + alpha_supply_line*(desired_supply_line - supply_line), where
#     supply_line is the pipeline (on-order + in-transit, i.e. everything
#     already placed but not yet received) and desired_supply_line =
#     lead_time * forecast (the pipeline you'd carry in steady state).
#
#     alpha_supply_line = 1 means fully crediting the pipeline - and at
#     alpha_stock=1, desired_stock=0, this reduces algebraically to exactly
#     naive_target(lead_time) - net_inventory, i.e. this policy at
#     alpha_supply_line=1 IS the naive NetUptoOrderingPolicy baseline below,
#     not just something similar to it:
#         order = forecast + 1*(0 - on_hand) + 1*(lead_time*forecast - supply_line)
#               = forecast*(1 + lead_time) - (on_hand + supply_line)
#               = naive_target(lead_time) - net_inventory
#     Sterman's central finding, replicated repeatedly (Croson & Donohue found
#     98% of 172 players underweight the supply line in the original study),
#     is that real players set alpha_supply_line well below 1 - they don't
#     fully believe an order already placed is "on the way," so a delayed
#     shipment reads as a real shortfall on top of what's already coming, and
#     they order more. Sweeping alpha_supply_line down from 1 isolates that
#     effect on its own, holding everything else at the naive baseline.
mutable struct AnchorAndAdjustOrderingPolicy <: InventoryOrderingPolicy
    alpha_stock::Float64
    alpha_supply_line::Float64
    desired_stock::Float64
end

function get_parameters(policy::AnchorAndAdjustOrderingPolicy)
    return [policy.alpha_stock, policy.alpha_supply_line, policy.desired_stock]
end

function set_parameters!(policy::AnchorAndAdjustOrderingPolicy, values::Array{Float64, 1})
    policy.alpha_stock = values[1]
    policy.alpha_supply_line = values[2]
    policy.desired_stock = values[3]
end

#=
get_order itself is correct - confirmed by calling it directly, standalone,
outside of place_orders: it returned exactly the value the naive base-stock
math predicts (target 30 - net_inventory 20 = 10). The bug is entirely in
how the installed package's place_orders reaches (or rather, doesn't reach)
this method during a real simulation run: the exact same call, made from
inside place_orders' hot loop instead of directly, silently behaves as if
get_order returned 0 every period, for every scenario, collapsing fill rate
to ~1%. That's consistent with a method-invalidation/specialization gap
between the package's precompiled place_orders and a get_order method added
from outside the module after the package was already precompiled, though
the exact mechanism isn't fully pinned down - see the place_orders override
below for the actual fix.
=#
function get_order(policy::AnchorAndAdjustOrderingPolicy, state::State, env::Env, location::Storage, lane::Lane, product::Product, time::Int64)::Int64
    forecast = MEAN_DEMAND
    on_hand = get_on_hand_inventory(state, location, product)
    net_inventory = get_net_inventory(state, location, product, time)
    supply_line = net_inventory - on_hand
    lead_time = lane.times[1]
    desired_supply_line = lead_time * forecast

    stock_adjustment = policy.alpha_stock * (policy.desired_stock - on_hand)
    supply_line_adjustment = policy.alpha_supply_line * (desired_supply_line - supply_line)

    order = forecast + stock_adjustment + supply_line_adjustment
    return max(0, round(Int, order))
end

#=
Overrides the installed package's place_orders(::State, ::Env, ::ConcreteNode,
::Product, ::Int, ::Array{OrderLine,1}) for Storage locations specifically
(Storage <: ConcreteNode, so this is a strict, unambiguous refinement of
that signature - it doesn't touch Customer's or Supplier's own place_orders
methods). The original hand-writes an inline isa-chain union split over the
built-in policy types as a performance optimization, with a generic `else`
branch that's supposed to reach any other policy type via ordinary dynamic
dispatch - in practice, for AnchorAndAdjustOrderingPolicy, it doesn't (see
above). This version always dispatches get_order generically - the same
calling pattern already confirmed to work correctly from outside the
package - which this small illustrative script doesn't need to avoid for
performance reasons, and adds an explicit check that fails loudly if
dispatch would silently resolve to the abstract InventoryOrderingPolicy
fallback (which just returns 0) instead of a real policy-specific method,
rather than ever silently treating an unrecognized policy as "order
nothing."
=#
function place_orders(state::State, env::Env, location::Storage, product::Product, time::Int, orders::Array{OrderLine, 1})
    empty!(orders)
    for trip in get_inbound_trips(env, location, time)
        policy = get(trip.policies, product, nothing)
        if !isnothing(policy)
            resolved = which(get_order, (typeof(policy), State, Env, typeof(location), Lane, Product, Int64))
            if resolved.sig.parameters[2] == InventoryOrderingPolicy
                error("place_orders: get_order for policy type $(typeof(policy)) resolves to the generic InventoryOrderingPolicy fallback (always returns 0), not a type-specific method - refusing to silently treat this as a zero order. Define get_order(::$(typeof(policy)), ::State, ::Env, ::ConcreteNode, ::Lane, ::Product, ::Int64) explicitly.")
            end
            quantity = Int(get_order(policy, state, env, location, trip.route, product, time))
            if quantity > 0
                minimum_quantity = trip.route.minimum_quantity
                if minimum_quantity > 0 && quantity < minimum_quantity
                    quantity = Int(ceil(minimum_quantity))
                end
                order = OrderLine(time, trip.route.origin, location, product, quantity, typemax(Int64), trip)
                push!(orders, order)
                push!(state.placed_orders, order)
                record_placement!(state, env, order)
                state.metrics.orders += quantity
            end
        end
    end
end

function run_anchor_and_adjust(alpha_supply_line, product, horizon, scenario_count, seed)
    policy2 = AnchorAndAdjustOrderingPolicy(1.0, alpha_supply_line, 0.0)
    policy3 = AnchorAndAdjustOrderingPolicy(1.0, alpha_supply_line, 0.0)
    policy4 = AnchorAndAdjustOrderingPolicy(1.0, alpha_supply_line, 0.0)
    policies = Dict((l2, product) => policy2, (l3, product) => policy3, (l4, product) => policy4)
    scenarios = build_scenarios(scenario_count, seed)
    states = [simulate(s, policies) for s in scenarios]
    return (
        aggregate = aggregate(states),
        bullwhip = bullwhip_ratios(states, LANES_NAMED, customer, product, horizon),
        inventory_cv = inventory_cv(states, STORAGES_NAMED, product, horizon),
        costs = cost_breakdown(states),
        backlog = backlog_summary(states, BACKLOG_NODES_NAMED, product, horizon),
        classic = classic_score(states, product, horizon),
    )
end

# --- Full metric bundle for a policy set already run to completion, reused
#     for the tuned-NetUptoOrderingPolicy and bigger-budget-optimizer runs
#     below so every configuration in this script is measured identically. ---
function full_metrics(states, product, horizon)
    return (
        aggregate = aggregate(states),
        bullwhip = bullwhip_ratios(states, LANES_NAMED, customer, product, horizon),
        inventory_cv = inventory_cv(states, STORAGES_NAMED, product, horizon),
        costs = cost_breakdown(states),
        backlog = backlog_summary(states, BACKLOG_NODES_NAMED, product, horizon),
        classic = classic_score(states, product, horizon),
    )
end

# --- Naive baseline: order-up-to a fixed "pipeline coverage, no safety stock"
#     target at each echelon (mean demand * (lead_time + 1)), never tuned. ---
naive_target(lead_time) = round(Int, MEAN_DEMAND * (lead_time + 1))

naive_policy2 = NetUptoOrderingPolicy(naive_target(2))  # wholesaler -> retailer
naive_policy3 = NetUptoOrderingPolicy(naive_target(2))  # factory -> wholesaler
naive_policy4 = NetUptoOrderingPolicy(naive_target(4))  # supplier -> factory
naive_policies = Dict((l2, product) => naive_policy2, (l3, product) => naive_policy3, (l4, product) => naive_policy4)

calibration_scenarios = build_scenarios(SCENARIO_COUNT, CALIBRATION_SEED)
naive_states = [simulate(s, naive_policies) for s in calibration_scenarios]
naive_result = aggregate(naive_states)
naive_result_scenario1 = (
    total_demand = get_total_demand(naive_states[1]),
    total_sales = get_total_sales(naive_states[1]),
    total_lost_sales = get_total_lost_sales(naive_states[1]),
)
naive_bullwhip = bullwhip_ratios(naive_states, LANES_NAMED, customer, product, HORIZON)
naive_inventory_cv = inventory_cv(naive_states, STORAGES_NAMED, product, HORIZON)
naive_costs = cost_breakdown(naive_states)
naive_backlog = backlog_summary(naive_states, BACKLOG_NODES_NAMED, product, HORIZON)
naive_classic = classic_score(naive_states, product, HORIZON)

# --- Sterman anchor-and-adjust sweep: alpha_supply_line=1.0 reproduces the
#     naive baseline exactly (see the policy's docstring above for the
#     algebra); sweeping it down isolates how much supply-line underweighting
#     alone - the specific, published, measured human bias - degrades things,
#     independent of any other change to the policy. ---
const SUPPLY_LINE_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]
anchor_adjust_results = Dict(
    string(w) => run_anchor_and_adjust(w, product, HORIZON, SCENARIO_COUNT, CALIBRATION_SEED)
    for w in SUPPLY_LINE_WEIGHTS
)

# Correctness check, not just documentation: alpha_supply_line=1.0 is
# algebraically identical to the naive baseline (see the policy's docstring),
# so if these two numbers disagree by more than float noise, get_order isn't
# dispatching to AnchorAndAdjustOrderingPolicy's actual method - this exact
# check would have caught the get_order dispatch-ambiguity bug immediately
# instead of requiring a manual read of results.json.
let
    check_fill_rate = anchor_adjust_results["1.0"].aggregate.fill_rate
    naive_fill_rate = naive_result.fill_rate
    agree = isapprox(check_fill_rate, naive_fill_rate; atol=1e-6)
    println("Sanity check: anchor_adjust(alpha_supply_line=1.0) should equal naive baseline exactly.")
    println("  anchor_adjust fill_rate=$(check_fill_rate)  naive fill_rate=$(naive_fill_rate)  agree=$(agree)")
    if !agree
        error("AnchorAndAdjustOrderingPolicy at alpha_supply_line=1.0 does not match the naive baseline - get_order is likely not dispatching correctly (see the dispatch-ambiguity note above the policy's get_order method).")
    end
end

# --- Optimized: BackwardCoverageOrderingPolicy at each upstream echelon,
#     tuned jointly by optimize!() against the same 30 calibration scenarios. ---
opt_policy2 = BackwardCoverageOrderingPolicy([0.0, 0.0])
opt_policy3 = BackwardCoverageOrderingPolicy([0.0, 0.0])
opt_policy4 = BackwardCoverageOrderingPolicy([0.0, 0.0])
opt_policies = Dict((l2, product) => opt_policy2, (l3, product) => opt_policy3, (l4, product) => opt_policy4)

# Rebuild the calibration scenarios with the exact same seed so optimize!
# (which mutates/resets state internally) sees an identical scenario set to
# the one the naive baseline above was evaluated on.
calibration_scenarios_for_opt = build_scenarios(SCENARIO_COUNT, CALIBRATION_SEED)
optimize!(opt_policies, calibration_scenarios_for_opt...; cost_function=metrics_cost_function, record_history=false)

optimized_in_sample_states = [simulate(s, opt_policies) for s in calibration_scenarios_for_opt]
optimized_in_sample = aggregate(optimized_in_sample_states)
optimized_in_sample_scenario1 = (
    total_demand = get_total_demand(optimized_in_sample_states[1]),
    total_sales = get_total_sales(optimized_in_sample_states[1]),
    total_lost_sales = get_total_lost_sales(optimized_in_sample_states[1]),
)
optimized_bullwhip = bullwhip_ratios(optimized_in_sample_states, LANES_NAMED, customer, product, HORIZON)
optimized_inventory_cv = inventory_cv(optimized_in_sample_states, STORAGES_NAMED, product, HORIZON)
optimized_costs = cost_breakdown(optimized_in_sample_states)
optimized_backlog = backlog_summary(optimized_in_sample_states, BACKLOG_NODES_NAMED, product, HORIZON)
optimized_classic = classic_score(optimized_in_sample_states, product, HORIZON)

# --- Follow-up 1: tune NetUptoOrderingPolicy's own (single) parameter too,
#     instead of comparing a *tuned* BackwardCoverageOrderingPolicy against
#     an *untuned* naive one. This is the fair, apples-to-apples comparison:
#     best-of-family vs. best-of-family. NetUptoOrderingPolicy has one
#     parameter per echelon and a monotonic, well-behaved cost surface (more
#     `upto` strictly trades lost sales for holding cost), so this should
#     converge easily regardless of the optimizer's own limitations. ---
tuned_naive_policy2 = NetUptoOrderingPolicy(0)
tuned_naive_policy3 = NetUptoOrderingPolicy(0)
tuned_naive_policy4 = NetUptoOrderingPolicy(0)
tuned_naive_policies = Dict((l2, product) => tuned_naive_policy2, (l3, product) => tuned_naive_policy3, (l4, product) => tuned_naive_policy4)
scenarios_for_tuned_naive = build_scenarios(SCENARIO_COUNT, CALIBRATION_SEED)
optimize!(tuned_naive_policies, scenarios_for_tuned_naive...; cost_function=metrics_cost_function, record_history=false)
tuned_naive_states = [simulate(s, tuned_naive_policies) for s in scenarios_for_tuned_naive]
tuned_naive_metrics = full_metrics(tuned_naive_states, product, HORIZON)
tuned_naive_targets = Dict(
    "l2_wholesaler_to_retailer" => tuned_naive_policy2.upto,
    "l3_factory_to_wholesaler" => tuned_naive_policy3.upto,
    "l4_supplier_to_factory" => tuned_naive_policy4.upto,
)

# --- Follow-up 2: rerun the original BackwardCoverageOrderingPolicy search
#     with 4x the evaluation budget (60000 vs. the default 15000) and 4x the
#     no-progress patience, same starting point, to check whether the
#     original result was actually converged or just cut off early. ---
big_opt_policy2 = BackwardCoverageOrderingPolicy([0.0, 0.0])
big_opt_policy3 = BackwardCoverageOrderingPolicy([0.0, 0.0])
big_opt_policy4 = BackwardCoverageOrderingPolicy([0.0, 0.0])
big_opt_policies = Dict((l2, product) => big_opt_policy2, (l3, product) => big_opt_policy3, (l4, product) => big_opt_policy4)
scenarios_for_big_opt = build_scenarios(SCENARIO_COUNT, CALIBRATION_SEED)
optimize!(big_opt_policies, scenarios_for_big_opt...; cost_function=metrics_cost_function, record_history=false,
          params=Dict(:MaxFuncEvals => 60000.0, :MaxStepsWithoutProgress => 6000.0))
big_opt_states = [simulate(s, big_opt_policies) for s in scenarios_for_big_opt]
big_opt_metrics = full_metrics(big_opt_states, product, HORIZON)
big_opt_cover = Dict(
    "l2_wholesaler_to_retailer" => big_opt_policy2.cover,
    "l3_factory_to_wholesaler" => big_opt_policy3.cover,
    "l4_supplier_to_factory" => big_opt_policy4.cover,
)

# Known-good values from test/policy-beergame-tests.jl's beer_game() test,
# for the exact same seed/config/policy family - printed as a sanity check,
# not asserted, since minor package-version drift could shift float results.
println("Sanity check vs. repo test/policy-beergame-tests.jl beer_game() (expected 103.0 / 1828.0 / 1931.0 for scenario 1):")
println("  got: lost_sales=$(optimized_in_sample_scenario1.total_lost_sales), sales=$(optimized_in_sample_scenario1.total_sales), demand=$(optimized_in_sample_scenario1.total_demand)")

# --- Generalization: same tuned policies, fresh out-of-sample scenarios. ---
holdout_scenarios = build_scenarios(SCENARIO_COUNT, HOLDOUT_SEED)
optimized_holdout_states = [simulate(s, opt_policies) for s in holdout_scenarios]
optimized_holdout = aggregate(optimized_holdout_states)

results = Dict(
    "horizon" => HORIZON,
    "scenario_count" => SCENARIO_COUNT,
    "calibration_seed" => CALIBRATION_SEED,
    "holdout_seed" => HOLDOUT_SEED,
    "mean_demand" => MEAN_DEMAND,
    "naive_targets" => Dict("l2_wholesaler_to_retailer" => naive_target(2), "l3_factory_to_wholesaler" => naive_target(2), "l4_supplier_to_factory" => naive_target(4)),
    "naive_aggregate" => naive_result,
    "naive_scenario1" => naive_result_scenario1,
    "naive_bullwhip_ratios" => naive_bullwhip,
    "naive_inventory_cv" => naive_inventory_cv,
    "naive_costs" => naive_costs,
    "naive_backlog" => naive_backlog,
    "naive_classic_score" => naive_classic,
    "optimized_in_sample_aggregate" => optimized_in_sample,
    "optimized_in_sample_scenario1" => optimized_in_sample_scenario1,
    "optimized_bullwhip_ratios" => optimized_bullwhip,
    "optimized_inventory_cv" => optimized_inventory_cv,
    "optimized_costs" => optimized_costs,
    "optimized_backlog" => optimized_backlog,
    "optimized_classic_score" => optimized_classic,
    "optimized_holdout_aggregate" => optimized_holdout,
    "tuned_policy_cover" => Dict(
        "l2_wholesaler_to_retailer" => opt_policy2.cover,
        "l3_factory_to_wholesaler" => opt_policy3.cover,
        "l4_supplier_to_factory" => opt_policy4.cover,
    ),
    "anchor_adjust_results" => anchor_adjust_results,
    "tuned_naive_metrics" => tuned_naive_metrics,
    "tuned_naive_targets" => tuned_naive_targets,
    "big_opt_metrics" => big_opt_metrics,
    "big_opt_cover" => big_opt_cover,
)

open(joinpath(@__DIR__, "results.json"), "w") do io
    JSON3.pretty(io, results)
end

println("\nWrote results.json")
println("Naive:            fill_rate=$(round(naive_result.fill_rate; digits=4))  lost_sales=$(round(naive_result.total_lost_sales; digits=1))")
println("Optimized (in):   fill_rate=$(round(optimized_in_sample.fill_rate; digits=4))  lost_sales=$(round(optimized_in_sample.total_lost_sales; digits=1))")
println("Optimized (hold): fill_rate=$(round(optimized_holdout.fill_rate; digits=4))  lost_sales=$(round(optimized_holdout.total_lost_sales; digits=1))")
println("\nBullwhip ratios (Var(orders)/Var(demand); >1 amplifies, <1 damps):")
println("  naive:     ", naive_bullwhip)
println("  optimized: ", optimized_bullwhip)
println("\nOn-hand inventory coefficient of variation by node (std/mean):")
println("  naive:     ", naive_inventory_cv)
println("  optimized: ", optimized_inventory_cv)
println("\nTotal cost (metrics_cost_function, the literal optimize! objective):")
println("  naive:     ", naive_costs)
println("  optimized: ", optimized_costs)
println("\nBacklog by layer (peak / ending, units, averaged across scenarios):")
println("  naive:     ", naive_backlog)
println("  optimized: ", optimized_backlog)
println("\nClassic beer-game score (holding + backlog cost per stage, retailer uses lost-sales proxy):")
println("  naive:     ", naive_classic)
println("  optimized: ", optimized_classic)
println("\nAnchor-and-adjust sweep (fill_rate, total_cost by alpha_supply_line):")
for w in SUPPLY_LINE_WEIGHTS
    r = anchor_adjust_results[string(w)]
    println("  alpha_supply_line=$(w): fill_rate=$(round(r.aggregate.fill_rate; digits=4))  total_cost=$(round(r.costs.total_cost; digits=1))")
end
println("\nTuned NetUptoOrderingPolicy (fair tuned-vs-tuned comparison):")
println("  targets: ", tuned_naive_targets)
println("  fill_rate=$(round(tuned_naive_metrics.aggregate.fill_rate; digits=4))  total_cost=$(round(tuned_naive_metrics.costs.total_cost; digits=1))")
println("\nBigger-budget BackwardCoverageOrderingPolicy rerun (60000 vs 15000 evals):")
println("  cover: ", big_opt_cover)
println("  fill_rate=$(round(big_opt_metrics.aggregate.fill_rate; digits=4))  total_cost=$(round(big_opt_metrics.costs.total_cost; digits=1))")
