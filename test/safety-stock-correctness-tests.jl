using Random
using Distributions: Normal, quantile, pdf

@testset "Safety stock correctness (closed-form validation)" begin

# Zero-external-dependency correctness benchmark for the upcoming simulation-calibrated
# buffer loop (see the plan's Section 5.1): a single-node Supplier -> Storage -> Customer
# chain, ordered with a plain base-stock (order-up-to) policy, has a known closed-form
# safety-stock answer from any standard inventory-management textbook (e.g. Silver, Pyke &
# Peterson, "Inventory and Production Management"): for periodic review with review period
# 1 and lead time L, the order-up-to level must cover demand over the protection period
# L+1, i.e. `order_up_to = mean_demand*(L+1) + z*sigma*sqrt(L+1)`. This is the target the
# calibration loop will later be checked against (does the loop converge to ~this answer
# starting from a bad guess), and on its own it's a correctness check that the simulator's
# order-up-to mechanics (NetUptoOrderingPolicy + lead-time handling) actually deliver the
# textbook-promised service level - no external published instance needed.
@test begin
    Random.seed!(20260708)

    horizon = 200
    lead_time = 3
    protection_period = lead_time + 1
    mean_demand = 20.0
    demand_std = 5.0
    target_service_level = 0.95
    scenario_count = 40

    z = quantile(Normal(), target_service_level)
    order_up_to = Int(round(mean_demand * protection_period + z * demand_std * sqrt(protection_period)))

    total_ordered = 0
    total_filled = 0

    for i in 1:scenario_count
        product = Product("p1")
        customer = Customer("c1")
        storage = Storage("s1"; initial_opened=true)
        add_product!(storage, product; unit_holding_cost=0.1)
        supplier = Supplier("sup1")
        add_product!(supplier, product; unit_cost=1.0)

        sc = SupplyChain(horizon)
        add_product!(sc, product)
        add_customer!(sc, customer)
        add_storage!(sc, storage)
        add_supplier!(sc, supplier)

        demand = [max(0.0, round(rand(Normal(mean_demand, demand_std)))) for _ in 1:horizon]
        add_demand!(sc, customer, product, demand; service_level=target_service_level, lost_sales_cost=1.0, sales_price=1.0)

        add_lane!(sc, Lane(storage, customer; unit_cost=0.0))
        supply_lane = Lane(supplier, storage; unit_cost=0.0, time=lead_time)
        add_lane!(sc, supply_lane)

        policy = NetUptoOrderingPolicy(order_up_to)
        state = simulate(sc, Dict((supply_lane, product) => policy))

        orders = filter(ol -> ol.destination == customer && ol.product == product, collect(Base.Iterators.flatten(state.historical_orders)))
        filled = filter(ol -> ol.destination == customer && ol.product == product, collect(Base.Iterators.flatten(state.historical_filled_orders)))

        total_ordered += sum(ol -> ol.quantity, orders; init=0)
        total_filled += sum(ol -> ol.quantity, filled; init=0)
    end

    achieved_fill_rate = total_filled / total_ordered
    println("Target service level: $target_service_level, achieved fill rate across $scenario_count scenarios: $achieved_fill_rate")

    # This isn't quite an apples-to-apples check: `order_up_to` is sized off a *cycle
    # service level* z-factor (probability of no stockout in the protection period), but
    # what's measured here is *fill rate* (fraction of ordered units actually filled) -
    # and those two are known to differ systematically, not just by simulation noise. The
    # standard order-fill-rate approximation for a periodic-review base-stock policy
    # (Silver, Pyke & Peterson, "Inventory and Production Management", ch. 7) is
    # `fill_rate ~ 1 - sigma_L * L(z) / Q`, where `L` is the standard normal loss function
    # and `Q` is the order quantity per review period (~mean_demand here, review period 1).
    # Plugging in these parameters (sigma_L = demand_std*sqrt(protection_period) = 10,
    # L(z=1.645) ~ 0.0208, Q ~ mean_demand = 20) predicts fill_rate ~ 1 - 10*0.0208/20 ~
    # 0.99 - a few points above the 95% cycle-service-level target, not a bug. So the
    # tolerance below is centered on that expected gap, not just on the target itself.
    expected_fill_rate = 1 - (demand_std * sqrt(protection_period)) * (pdf(Normal(), z) - z * (1 - target_service_level)) / mean_demand
    abs(achieved_fill_rate - expected_fill_rate) < 0.03
end

end
