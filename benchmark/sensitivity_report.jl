#=
Runs sensitivity_analysis (see Optimization.jl's sobol_indices/sensitivity_analysis)
against the same two problems benchmark/compare_optimizers.jl uses - "newsvendor"
(2 parameters) and "beer_game" (6 parameters) - to get real Sobol S1/ST numbers
instead of sensitivity-analysis-tests.jl's :samples=>4 smoke test, which is sized
to be fast, not to be informative.

The question this answers is different from compare_optimizers.jl's "which search
algorithm wins?": "within a policy's parameters, which ones actually move the cost,
and which are along for the ride?" That matters for two things:
- Triage before spending an optimize! budget: a parameter with S1/ST near 0 across
  a wide box isn't worth tuning finely.
- Understanding *why* an optimizer struggled - e.g. :custom's differential
  evolution population being spread across parameters that barely affect cost
  wastes budget compared to concentrating on the ones that do.
- Sizing a future surrogate/gradient-based refinement step: fewer
  effective dimensions (parameters with real S1) means an easier surrogate to
  fit well.

`ST >> S1` for a parameter flags that it mostly matters through interaction with
another parameter, not on its own - `metrics_cost_function`'s inputs (holding
costs, lost sales, trip costs across lead-time-linked echelons) are exactly the
kind of coupled system where that's plausible for beer_game's 6 parameters,
unlike newsvendor's single-lane 2-parameter problem.

Run with:
    julia --project=benchmark benchmark/sensitivity_report.jl

Sample count (and thus cost: `samples * (parameters + 2)` simulations per
problem) comes from the SENSITIVITY_SAMPLES env var, default 200 - much
smaller than sensitivity_analysis's own default of 1000, chosen to keep this
in the same runtime ballpark as compare_optimizers.jl's beer_game comparison
rather than sensitivity_analysis's default (sized for a thorough one-off
analysis, not routine CI).

Also runnable via .github/workflows/sensitivity-report.yml.
=#

using Random
using Distributions: Poisson, Normal
using SupplyChainModeling
using SupplyChainSimulation

function sensitivity_report_params()
    samples = parse(Int, get(ENV, "SENSITIVITY_SAMPLES", "200"))
    return (; samples)
end

# =============================================================================
# Problem 1: newsvendor (mirrors compare_optimizers.jl's build_scenario)
# =============================================================================
const HORIZON = 100
const LEAD_TIME = 3
const TARGET_SERVICE_LEVEL = 0.95
const HOLDING_COST = 0.1
const LOST_SALES_COST = HOLDING_COST * TARGET_SERVICE_LEVEL / (1 - TARGET_SERVICE_LEVEL)
const TRAINING_REGIMES = [(10.0, 2.0), (20.0, 5.0), (30.0, 7.0), (15.0, 6.0)]
const TRAINING_REPLICATIONS = 4

function build_scenario(mean_demand, demand_std)
    product = Product("p1")
    customer = Customer("c1")
    storage = Storage("s1"; initial_opened=true)
    add_product!(storage, product; unit_holding_cost=HOLDING_COST)
    supplier = Supplier("sup1")
    add_product!(supplier, product; unit_cost=1.0)

    sc = SupplyChain(HORIZON)
    add_product!(sc, product)
    add_customer!(sc, customer)
    add_storage!(sc, storage)
    add_supplier!(sc, supplier)

    demand = [max(0.0, round(rand(Normal(mean_demand, demand_std)))) for _ in 1:HORIZON]
    add_demand!(sc, customer, product, demand; lost_sales_cost=LOST_SALES_COST, sales_price=0.0)

    add_lane!(sc, Lane(storage, customer; unit_cost=0.0))
    supply_lane = Lane(supplier, storage; unit_cost=0.0, time=LEAD_TIME)
    add_lane!(sc, supply_lane)

    return sc, supply_lane, product
end

newsvendor_cost(state) = get_total_lost_sales(state) + get_total_holding_costs(state)

function run_newsvendor_sensitivity(samples::Int)
    Random.seed!(3000)
    scenarios = SupplyChain[]
    supply_lane = nothing
    product = nothing
    for (mean_demand, demand_std) in TRAINING_REGIMES, _ in 1:TRAINING_REPLICATIONS
        sc, lane, p = build_scenario(mean_demand, demand_std)
        push!(scenarios, sc)
        supply_lane = lane
        product = p
    end

    policy = BackwardCoverageOrderingPolicy([0.0, 0.0])
    lane_policies = Dict((supply_lane, product) => policy)

    return sensitivity_analysis(lane_policies, scenarios...;
        cost_function=newsvendor_cost, record_history=false,
        options=Dict{Symbol, Any}(:lower => 0.0, :upper => 150.0, :samples => samples))
end

# =============================================================================
# Problem 2: beer_game (mirrors compare_optimizers.jl's build_beergame_scenario)
# =============================================================================
const BEERGAME_HORIZON = 200
const BEERGAME_HOLDING_COST = 0.1
const BEERGAME_TRAINING_COUNT = 20

function build_beergame_scenario()
    product = Product("beer")
    customer = Customer("customer")
    retailer = Storage("retailer")
    add_product!(retailer, product; unit_holding_cost=BEERGAME_HOLDING_COST, initial_inventory=20)
    wholesaler = Storage("wholesaler")
    add_product!(wholesaler, product; unit_holding_cost=BEERGAME_HOLDING_COST, initial_inventory=20)
    factory = Storage("factory")
    add_product!(factory, product; unit_holding_cost=BEERGAME_HOLDING_COST, initial_inventory=20)
    supplier = Supplier("supplier")

    sc = SupplyChain(BEERGAME_HORIZON)
    add_supplier!(sc, supplier)
    add_storage!(sc, retailer)
    add_storage!(sc, wholesaler)
    add_storage!(sc, factory)
    add_customer!(sc, customer)
    add_product!(sc, product)

    l1 = Lane(retailer, customer; unit_cost=0.0)
    l2 = Lane(wholesaler, retailer; unit_cost=0.0, time=2)
    l3 = Lane(factory, wholesaler; unit_cost=0.0, time=2)
    l4 = Lane(supplier, factory; unit_cost=0.0, time=4)
    add_lane!(sc, l1)
    add_lane!(sc, l2)
    add_lane!(sc, l3)
    add_lane!(sc, l4)

    demand = rand(Poisson(10), BEERGAME_HORIZON) * 1.0
    add_demand!(sc, customer, product, demand; sales_price=1.0, lost_sales_cost=1.0)

    return sc, l2, l3, l4, product
end

function run_beergame_sensitivity(samples::Int)
    Random.seed!(4000)
    scenarios = SupplyChain[]
    l2 = l3 = l4 = nothing
    product = nothing
    for _ in 1:BEERGAME_TRAINING_COUNT
        sc, l2_, l3_, l4_, p = build_beergame_scenario()
        push!(scenarios, sc)
        l2, l3, l4, product = l2_, l3_, l4_, p
    end

    policy2 = BackwardCoverageOrderingPolicy([0.0, 0.0])
    policy3 = BackwardCoverageOrderingPolicy([0.0, 0.0])
    policy4 = BackwardCoverageOrderingPolicy([0.0, 0.0])
    lane_policies = Dict((l2, product) => policy2, (l3, product) => policy3, (l4, product) => policy4)

    return sensitivity_analysis(lane_policies, scenarios...;
        cost_function=metrics_cost_function, record_history=false,
        options=Dict{Symbol, Any}(:lower => 0.0, :upper => 150.0, :samples => samples))
end

function report_table(problem_name::String, result)
    order = sortperm(result.ST; rev=true)
    lines = String[]
    push!(lines, "## $problem_name")
    push!(lines, "")
    push!(lines, "| parameter | S1 (first-order) | ST (total-order) | ST - S1 (interaction signal) |")
    push!(lines, "|---|---|---|---|")
    for i in order
        push!(lines, "| $(result.parameter_labels[i]) | $(round(result.S1[i], digits=4)) | $(round(result.ST[i], digits=4)) | $(round(result.ST[i] - result.S1[i], digits=4)) |")
    end
    return lines
end

function main()
    (; samples) = sensitivity_report_params()
    println("Sobol sensitivity analysis: samples=$samples (=> $samples * (parameters+2) simulations per problem)\n")

    println("=== newsvendor ===")
    newsvendor_result = run_newsvendor_sensitivity(samples)
    for (label, s1, st) in zip(newsvendor_result.parameter_labels, newsvendor_result.S1, newsvendor_result.ST)
        println("newsvendor $label S1=$(round(s1, digits=4)) ST=$(round(st, digits=4))")
    end

    println("\n=== beer_game ===")
    beergame_result = run_beergame_sensitivity(samples)
    for (label, s1, st) in zip(beergame_result.parameter_labels, beergame_result.S1, beergame_result.ST)
        println("beer_game $label S1=$(round(s1, digits=4)) ST=$(round(st, digits=4))")
    end

    summary = join(vcat(report_table("newsvendor", newsvendor_result), [""], report_table("beer_game", beergame_result)), "\n")
    println("\n" * summary)

    summary_path = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
    if !isnothing(summary_path)
        open(summary_path, "a") do io
            println(io, summary)
        end
    end
end

main()
