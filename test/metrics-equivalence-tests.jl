using Random

# One-time equivalence test: while state.metrics (SimMetrics, updated inline
# at fill/drop/placement sites) and the historical_* arrays (scanned after
# the fact by Reporting.jl's get_total_* functions) both exist, every run
# below is simulated with the default record_history=true so both mechanisms
# are populated, and asserts they agree to the last unit/cent. This is what
# justifies trusting state.metrics alone (e.g. via metrics_cost_function)
# once a caller opts in to running optimize! with record_history=false.
function assert_metrics_match_history(final_state)
    metrics = get_metrics(final_state)

    @test metrics.sales ≈ get_total_sales(final_state)
    @test metrics.lost_sales ≈ get_total_lost_sales(final_state)
    @test metrics.holding_costs ≈ get_total_holding_costs(final_state)
    @test metrics.overflow_costs ≈ get_total_overflow_costs(final_state)
    @test metrics.trip_unit_costs ≈ get_total_trip_unit_costs(final_state)
    @test metrics.trip_fixed_costs ≈ get_total_trip_fixed_costs(final_state)
    @test metrics.orders ≈ get_total_orders(final_state)
    @test metrics.demand ≈ get_total_demand(final_state)
end

@testset "SimMetrics vs history-scan equivalence" begin
    @testset "random demand, various horizons and policies" begin
        Random.seed!(42)

        for horizon in [1, 2, 5, 30]
            product = Product("product")

            supplier = Supplier("supplier")
            storage = Storage("storage")
            add_product!(storage, product; unit_holding_cost=0.7)
            customer = Customer("customer")

            l1 = Lane(storage, customer; unit_cost=0.3)
            l2 = Lane(supplier, storage; unit_cost=0.1, fixed_cost=15, time=2)

            network = SupplyChain(horizon)
            add_supplier!(network, supplier)
            add_storage!(network, storage)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l1)
            add_lane!(network, l2)
            add_demand!(network, customer, product, rand(Poisson(8), horizon) * 1.0; sales_price=3.0, lost_sales_cost=2.0)

            for policy in [NetSSOrderingPolicy(5, 20), OnHandUptoOrderingPolicy(15), ForwardCoverageOrderingPolicy(2.0)]
                policies = Dict((l2, product) => policy)
                final_state = simulate(network, policies)
                assert_metrics_match_history(final_state)
            end
        end
    end

    @testset "lost sales dropped mid-run vs. lost sales still pending at horizon end" begin
        # Deliberately starved storage (no replenishment policy at all) so
        # every period's customer demand goes unfulfilled - some of it
        # expiring via the explicit due_date < time path in send_inventory!,
        # and, crucially, the very last period's demand never reaching that
        # path at all (the loop ends at time == horizon, so due_date < time
        # never becomes true for an order created at time == horizon). Both
        # cases must be reflected in metrics.lost_sales - see
        # flush_pending_as_lost! in Simulation.jl.
        for horizon in [1, 3, 10]
            product = Product("product")
            customer = Customer("c")
            storage = Storage("s")
            add_product!(storage, product)
            l = Lane(storage, customer; unit_cost=0)

            network = SupplyChain(horizon)
            add_storage!(network, storage)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l)
            add_demand!(network, customer, product, fill(7.0, horizon); sales_price=2.0, lost_sales_cost=5.0)

            policies = Dict{Tuple{Lane, Product}, InventoryOrderingPolicy}()
            final_state = simulate(network, policies)

            @test get_total_lost_sales(final_state) == 7.0 * 5.0 * horizon
            assert_metrics_match_history(final_state)
        end
    end

    @testset "overflow costs via a live (not pre-scheduled) zero-lead-time shipment" begin
        # 3-node chain (storage2 -> storage -> customer) with a zero lead
        # time on the second leg, so the overflow-triggering arrival at
        # storage comes from a shipment actually sent during the run (in
        # send_inventory!'s loop, picked up by storage's second
        # receive_inventory! call that period), rather than from
        # initial_arrivals being pre-loaded before period 1 the way
        # constraint-enforcement-tests.jl's overflow case does it.
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product; maximum_units=5, overflow_unit_cost=2.0)
        storage2 = Storage("s2")
        add_product!(storage2, product; initial_inventory=10)
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0, time=0)

        network = SupplyChain(2)
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)
        add_demand!(network, customer, product, [0.0, 0.0]; sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict((l2, product) => OnHandUptoOrderingPolicy(10))
        final_state = simulate(network, policies)

        @test get_total_overflow_costs(final_state) > 0
        assert_metrics_match_history(final_state)
    end

    @testset "trip fixed cost is deduped across order lines sharing a departure" begin
        # Two products moving on the same lane/departure within a period
        # share the exact same Trip object (Env.departures holds one Trip
        # per (lane, period), not per product) - get_total_trip_fixed_costs
        # charges that trip's fixed cost once, and metrics.trip_fixed_costs
        # must match, not double-charge it.
        productA = Product("A")
        productB = Product("B")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, productA)
        add_product!(storage, productB)
        customer = Customer("customer")

        l1 = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=0.5, fixed_cost=25)

        horizon = 4
        network = SupplyChain(horizon)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, productA)
        add_product!(network, productB)
        add_lane!(network, l1)
        add_lane!(network, l2)
        add_demand!(network, customer, productA, fill(3.0, horizon); sales_price=1.0, lost_sales_cost=1.0)
        add_demand!(network, customer, productB, fill(3.0, horizon); sales_price=1.0, lost_sales_cost=1.0)

        policy = NetSSOrderingPolicy(2, 15)
        policies = Dict((l2, productA) => policy, (l2, productB) => policy)
        final_state = simulate(network, policies)

        assert_metrics_match_history(final_state)
    end

    @testset "batch of initial states, as optimize! evaluates them" begin
        Random.seed!(7)
        horizon = 15
        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.2)
        customer = Customer("customer")

        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage; fixed_cost=10, time=1)

        n() = begin
            network = SupplyChain(horizon)
            add_supplier!(network, supplier)
            add_storage!(network, storage)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l1)
            add_lane!(network, l2)
            add_demand!(network, customer, product, rand(Poisson(6), horizon) * 1.0; sales_price=1.5, lost_sales_cost=1.0)
            return network
        end

        policy = NetSSOrderingPolicy(3, 12)
        policies = Dict((l2, product) => policy)

        for network in [n() for _ in 1:5]
            final_state = simulate(network, policies)
            assert_metrics_match_history(final_state)
        end
    end

    @testset "optimize!'s default cost_function/record_history are unchanged; metrics_cost_function is opt-in" begin
        Random.seed!(11)
        horizon = 10
        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)
        customer = Customer("customer")

        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage; fixed_cost=5, time=1)

        n() = begin
            network = SupplyChain(horizon)
            add_supplier!(network, supplier)
            add_storage!(network, storage)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l1)
            add_lane!(network, l2)
            add_demand!(network, customer, product, rand(Poisson(5), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)
            return network
        end

        policy = NetSSOrderingPolicy(0, 0)
        policies = Dict((l2, product) => policy)
        initial_states = [n() for _ in 1:3]

        # Calling optimize! with no cost_function/record_history override must
        # keep recording real per-period history for every trial simulation,
        # exactly as it always did (this is what makes optimize!'s existing
        # default byte-for-byte unaffected by SimMetrics/record_history
        # existing at all - see metrics_cost_function's docstring).
        # historical_orders always ends up with exactly horizon+1 entries when
        # record_history is true (one from the initial snapshot_state!(state,
        # 0, ...) plus one per period): a deterministic invariant, unlike
        # "were any orders placed", which would depend on the random demand
        # draws and the optimizer's search.
        history_lengths = Int[]
        optimize!(policies, initial_states...;
                  params=Dict(:MaxFuncEvals => 10.0, :MaxStepsWithoutProgress => 10.0),
                  cost_function=s -> begin
                      push!(history_lengths, length(s.historical_orders))
                      get_total_lost_sales(s) + 0.001 * get_total_orders(s)
                  end)
        @test !isempty(history_lengths)
        @test all(==(horizon + 1), history_lengths)

        # The opt-in fast path - metrics_cost_function with record_history=false
        # - must also run to completion without erroring.
        optimize!(policies, initial_states...;
                  params=Dict(:MaxFuncEvals => 30.0, :MaxStepsWithoutProgress => 30.0),
                  cost_function=metrics_cost_function, record_history=false)
    end

    @testset "record_history=false rejects BackwardCoverageOrderingPolicy" begin
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(storage, customer; unit_cost=0)

        network = SupplyChain(3)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_demand!(network, customer, product, [1.0, 1.0, 1.0]; sales_price=1.0, lost_sales_cost=1.0)

        initial_state = State(network)
        policies = Dict((l, product) => BackwardCoverageOrderingPolicy([1.0, 1.0]))

        @test_throws ArgumentError Env(network, [initial_state], policies; record_history=false)
        # But it's fine (and the default) with history recording left on.
        @test Env(network, [initial_state], policies).record_history == true
    end
end
