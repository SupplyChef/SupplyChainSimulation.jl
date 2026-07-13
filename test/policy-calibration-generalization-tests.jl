using Random
using Distributions: Normal

@testset "Policy calibration generalizes across demand regimes" begin

# Validates the reframed design in the plan's Section 5.1: calibrate a *reactive* policy
# (BackwardCoverageOrderingPolicy, which recomputes its order target every period from
# recent order history, rather than a static frozen number) via simulation-optimization
# across a *diverse training ensemble* spanning several demand regimes, then check it
# still holds the target service level on a *held-out* ensemble of different regimes it
# was never tuned on - the property a static safety-stock buffer can't offer, since that
# would need to be recomputed whenever the regime changes.
#
# The closed-form target here is the classic newsvendor critical ratio (see e.g. Nahmias,
# "Production and Operations Analysis", or Silver/Pyke/Peterson): for a single-stage
# system minimizing (lost_sales_cost * lost_units + holding_cost * held_units), the
# cost-optimal fill rate is critical_ratio = lost_sales_cost / (lost_sales_cost +
# holding_cost). We size lost_sales_cost so that ratio equals our target service level,
# use exactly that (lost sales + holding cost) sum as the calibration objective, and check
# the calibrated policy's *achieved* fill rate lands near the target - on data it wasn't
# calibrated on.
@test begin
    Random.seed!(20260709)

    horizon = 100
    lead_time = 3
    target_service_level = 0.95
    holding_cost = 0.1
    # critical_ratio = lost_sales_cost / (lost_sales_cost + holding_cost) = target_service_level
    lost_sales_cost = holding_cost * target_service_level / (1 - target_service_level)

    function build_scenario(mean_demand, demand_std)
        product = Product("p1")
        customer = Customer("c1")
        storage = Storage("s1"; initial_opened=true)
        add_product!(storage, product; unit_holding_cost=holding_cost)
        supplier = Supplier("sup1")
        add_product!(supplier, product; unit_cost=1.0)

        sc = SupplyChain(horizon)
        add_product!(sc, product)
        add_customer!(sc, customer)
        add_storage!(sc, storage)
        add_supplier!(sc, supplier)

        demand = [max(0.0, round(rand(Normal(mean_demand, demand_std)))) for _ in 1:horizon]
        add_demand!(sc, customer, product, demand; lost_sales_cost=lost_sales_cost, sales_price=0.0)

        add_lane!(sc, Lane(storage, customer; unit_cost=0.0))
        supply_lane = Lane(supplier, storage; unit_cost=0.0, time=lead_time)
        add_lane!(sc, supply_lane)

        return sc, supply_lane, product
    end

    # Training ensemble: four distinct demand regimes (different means and variances, not
    # just different random seeds of one distribution), several replications each.
    training_regimes = [(10.0, 2.0), (20.0, 5.0), (30.0, 7.0), (15.0, 6.0)]
    replications_per_training_regime = 4

    training_scenarios = []
    supply_lane = nothing
    product = nothing
    for (mean_demand, demand_std) in training_regimes, _ in 1:replications_per_training_regime
        sc, lane, p = build_scenario(mean_demand, demand_std)
        push!(training_scenarios, sc)
        supply_lane = lane
        product = p
    end

    policy = BackwardCoverageOrderingPolicy([0.0, 0.0])
    policies = Dict((supply_lane, product) => policy)

    newsvendor_cost(state) = get_total_lost_sales(state) + get_total_holding_costs(state)

    optimize!(policies, training_scenarios...;
              params=Dict(:MaxFuncEvals => 3000.0, :MaxStepsWithoutProgress => 500.0),
              cost_function=newsvendor_cost)

    # Held-out ensemble: regimes *not* seen during calibration (interpolated between the
    # training levels), fresh seeds - the calibrated policy is applied with no further
    # tuning.
    held_out_regimes = [(25.0, 6.0), (12.0, 4.0)]
    replications_per_held_out_regime = 5

    total_ordered = 0
    total_filled = 0
    for (mean_demand, demand_std) in held_out_regimes, _ in 1:replications_per_held_out_regime
        sc, lane, p = build_scenario(mean_demand, demand_std)
        held_out_policies = Dict((lane, p) => policy)
        state = simulate(sc, held_out_policies)

        orders = filter(ol -> ol.destination isa Customer && ol.product == p, collect(Base.Iterators.flatten(state.historical_orders)))
        filled = filter(ol -> ol.destination isa Customer && ol.product == p, collect(Base.Iterators.flatten(state.historical_filled_orders)))

        total_ordered += sum(ol -> ol.quantity, orders; init=0)
        total_filled += sum(ol -> ol.quantity, filled; init=0)
    end

    achieved_fill_rate = total_filled / total_ordered
    println("Target service level (critical ratio): $target_service_level, achieved fill rate on held-out regimes: $achieved_fill_rate, calibrated cover: $(policy.cover)")

    # Looser tolerance than the single-regime closed-form test: this is a harder task
    # (one policy, tuned on a finite budget across four different regimes, applied
    # without retuning to two regimes it never saw), so a wider band is the honest bar.
    abs(achieved_fill_rate - target_service_level) < 0.08
end

end
