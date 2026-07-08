using Random
using Statistics
using Distributions: Normal, quantile

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

    # A buffer sized exactly at the textbook formula should land close to the target
    # service level - not exactly (it's a finite ensemble of a stochastic system, and the
    # normal approximation to the demand distribution isn't exact for the clamped-at-zero
    # demand used here), but within a few points, not wildly off.
    abs(achieved_fill_rate - target_service_level) < 0.03
end

end
