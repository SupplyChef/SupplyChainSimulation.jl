using Distributions: DiscreteUniform

@testset "Lead-time scenarios" begin
    # supplier -> storage (lane "ls", nominal lead time 2) -> customer. Orders of 10 at periods 1 and 2.
    function build_network(; lead_times=nothing, horizon=6)
        product = Product("product")
        supplier = Supplier("supplier")
        storage = Storage("storage")
        customer = Customer("customer")
        add_product!(storage, product)
        ls = Lane(supplier, storage; id="ls", unit_cost=0, time=2, lead_times=lead_times)
        lc = Lane(storage, customer; id="lc", unit_cost=0)
        network = SupplyChain(horizon)
        add_product!(network, product)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_lane!(network, ls)
        add_lane!(network, lc)
        add_demand!(network, customer, product, zeros(horizon); sales_price=1.0, lost_sales_cost=1.0)
        return network, ls, storage, product
    end
    on_hand_history(state, storage, product) = [get(h, (storage, product), 0) for h in state.historical_on_hand]
    orders = [10, 10, 0, 0, 0, 0]

    # Nominal: both orders take 2 periods (arrive at 3 and 4).
    @test begin
        network, ls, storage, product = build_network()
        state = simulate(network, Dict((ls, product) => QuantityOrderingPolicy(orders)))
        on_hand_history(state, storage, product)[2:end] == [0, 0, 10, 20, 20, 20]
    end

    # Lead time by departure period: the order placed at 1 takes 4 periods (arrives at 5), the one at 2
    # takes 1 (arrives at 3), so the later shipment overtakes the earlier one.
    @test begin
        network, ls, storage, product = build_network(lead_times=[4, 1, 1, 1, 1, 1])
        state = simulate(network, Dict((ls, product) => QuantityOrderingPolicy(orders)))
        on_hand_history(state, storage, product)[2:end] == [0, 0, 10, 10, 20, 20]
    end

    # A shipment that would arrive after the horizon never arrives.
    @test begin
        network, ls, storage, product = build_network(lead_times=[7, 1, 1, 1, 1, 1])
        state = simulate(network, Dict((ls, product) => QuantityOrderingPolicy(orders)))
        on_hand_history(state, storage, product)[end] == 10
    end

    # lead_times equal to the nominal time behave exactly like having none.
    @test begin
        network, ls, storage, product = build_network(lead_times=[2, 2, 2, 2, 2, 2])
        state = simulate(network, Dict((ls, product) => QuantityOrderingPolicy(orders)))
        on_hand_history(state, storage, product)[2:end] == [0, 0, 10, 20, 20, 20]
    end

    # sample_scenarios
    base, ls, storage, product = build_network()
    spec = Dict("ls" => iid_lead_times(DiscreteUniform(1, 4)))

    scenarios = sample_scenarios(base, 5; seed=7, lead_times=spec)
    @test length(scenarios) == 5
    # Reproducible, and scenario i does not depend on how many are requested.
    @test [s.lanes[1].lead_times for s in sample_scenarios(base, 3; seed=7, lead_times=spec)] == [s.lanes[1].lead_times for s in scenarios[1:3]]
    @test scenarios[1].lanes[1].lead_times != scenarios[2].lanes[1].lead_times
    @test sample_scenarios(base, 1; seed=8, lead_times=spec)[1].lanes[1].lead_times != scenarios[1].lanes[1].lead_times
    # Draws are valid: one per period, within the distribution's support.
    @test all(s -> length(s.lanes[1].lead_times[1]) == 6 && all(x -> 1 <= x <= 4, s.lanes[1].lead_times[1]), scenarios)
    # The base is untouched, other lanes are shared, and a lane is the same lane in every scenario.
    @test base.lanes[1].lead_times === nothing && all(s -> s.lanes[2].lead_times === nothing, scenarios)
    @test all(s -> s.lanes[1] == base.lanes[1] && s.lanes[2] == base.lanes[2], scenarios)
    # So policies keyed by the base lane drive every scenario.
    @test all(scenarios) do s
        state = simulate(s, Dict((ls, product) => QuantityOrderingPolicy(orders)))
        state isa State
    end
    # constant_lead_time: one draw per scenario, used for every period.
    @test all(s -> length(unique(s.lanes[1].lead_times[1])) == 1,
              sample_scenarios(base, 5; seed=1, lead_times=Dict("ls" => constant_lead_time(DiscreteUniform(1, 4)))))
    # A custom sampler plugs in the same way.
    @test sample_scenarios(base, 2; lead_times=Dict("ls" => (rng, horizon, nominal) -> fill(nominal + 1, horizon)))[2].lanes[1].lead_times == [fill(3, 6)]
    # Errors.
    @test_throws ArgumentError sample_scenarios(base, 2; lead_times=Dict("nope" => iid_lead_times(DiscreteUniform(1, 4))))
    @test_throws ArgumentError sample_scenarios(base, 0; lead_times=spec)
    @test_throws ArgumentError sample_scenarios(base, 1; lead_times=Dict("ls" => (rng, horizon, nominal) -> [1, 2]))
end
