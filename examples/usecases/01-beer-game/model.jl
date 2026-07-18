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
using SupplyChainModeling, SupplyChainSimulation, Distributions, Random, JSON3

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
    "optimized_in_sample_aggregate" => optimized_in_sample,
    "optimized_in_sample_scenario1" => optimized_in_sample_scenario1,
    "optimized_holdout_aggregate" => optimized_holdout,
    "tuned_policy_cover" => Dict(
        "l2_wholesaler_to_retailer" => opt_policy2.cover,
        "l3_factory_to_wholesaler" => opt_policy3.cover,
        "l4_supplier_to_factory" => opt_policy4.cover,
    ),
)

open(joinpath(@__DIR__, "results.json"), "w") do io
    JSON3.pretty(io, results)
end

println("\nWrote results.json")
println("Naive:            fill_rate=$(round(naive_result.fill_rate; digits=4))  lost_sales=$(round(naive_result.total_lost_sales; digits=1))")
println("Optimized (in):   fill_rate=$(round(optimized_in_sample.fill_rate; digits=4))  lost_sales=$(round(optimized_in_sample.total_lost_sales; digits=1))")
println("Optimized (hold): fill_rate=$(round(optimized_holdout.fill_rate; digits=4))  lost_sales=$(round(optimized_holdout.total_lost_sales; digits=1))")
