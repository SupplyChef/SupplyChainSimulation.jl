using Random

@testset "Tariffs" begin
    @testset "Supplier-origin lane: static declared value (supplier's own unit_cost)" begin
        # Mirrors SupplyChainOptimization.jl's own worked example: 100 units
        # bought at unit_cost=10 from a CN supplier, tariffed 20% into the US
        # -> 0.2 * 10.0 * 100 = 200.
        product = Product("product")

        supplier = Supplier("supplier", Location(0.0, 0.0; country="CN"))
        add_product!(supplier, product; unit_cost=10.0)
        storage = Storage("storage", Location(0.0, 0.0; country="US"))
        customer = Customer("customer")

        l1 = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=0)

        network = SupplyChain(1)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)
        add_demand!(network, customer, product, [0.0]; sales_price=1.0, lost_sales_cost=1.0)
        add_tariff!(network, Tariff("CN", "US", 0.20))

        policies = Dict((l2, product) => QuantityOrderingPolicy([100]))
        final_state = simulate(network, policies)

        expected = 0.20 * 10.0 * 100
        @test get_total_tariff_costs(final_state) ≈ expected
        @test final_state.metrics.tariff_costs ≈ expected
    end

    @testset "no Tariff registered -> zero tariff cost, untouched by tariff-relevant bookkeeping" begin
        product = Product("product")

        supplier = Supplier("supplier", Location(0.0, 0.0; country="CN"))
        add_product!(supplier, product; unit_cost=10.0)
        storage = Storage("storage", Location(0.0, 0.0; country="US"))
        customer = Customer("customer")

        l1 = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=0)

        network = SupplyChain(1)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)
        add_demand!(network, customer, product, [0.0]; sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict((l2, product) => QuantityOrderingPolicy([100]))
        final_state = simulate(network, policies)

        @test final_state.tariff_context.needs_tracking == false
        @test get_total_tariff_costs(final_state) == 0.0
        @test final_state.metrics.tariff_costs == 0.0
    end

    @testset "Storage-origin re-export: blended cohort, one country per supplier, priced against each's average unit_cost" begin
        # storage (US) receives from two suppliers in the same period - a CN
        # supplier (unit_cost=8) and a DE supplier (unit_cost=12) - then
        # re-exports the full blended 150 units to a customer in MX in the
        # same period (see Simulation.jl's phase ordering: a Storage's
        # receive_inventory! and send_inventory! for a period both run before
        # the next location's turn). Tariff cost must split by cohort:
        # CN's 100 units at 15% and DE's 50 units at 10%, not a blended
        # average rate.
        product = Product("product")

        supplier_cn = Supplier("supplier_cn", Location(0.0, 0.0; country="CN"))
        add_product!(supplier_cn, product; unit_cost=8.0)
        supplier_de = Supplier("supplier_de", Location(0.0, 0.0; country="DE"))
        add_product!(supplier_de, product; unit_cost=12.0)
        storage = Storage("storage", Location(0.0, 0.0; country="US"))
        customer = Customer("customer", Location(0.0, 0.0; country="MX"))

        l_cn = Lane(supplier_cn, storage; unit_cost=0)
        l_de = Lane(supplier_de, storage; unit_cost=0)
        l_out = Lane(storage, customer; unit_cost=0)

        network = SupplyChain(1)
        add_supplier!(network, supplier_cn)
        add_supplier!(network, supplier_de)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l_cn)
        add_lane!(network, l_de)
        add_lane!(network, l_out)
        add_demand!(network, customer, product, [150.0]; sales_price=1.0, lost_sales_cost=1.0)
        add_tariff!(network, Tariff("CN", "MX", 0.15))
        add_tariff!(network, Tariff("DE", "MX", 0.10))

        policies = Dict((l_cn, product) => QuantityOrderingPolicy([100]),
                         (l_de, product) => QuantityOrderingPolicy([50]))
        final_state = simulate(network, policies)

        expected = 0.15 * 8.0 * 100 + 0.10 * 12.0 * 50
        @test get_total_tariff_costs(final_state) ≈ expected
        @test final_state.metrics.tariff_costs ≈ expected
    end

    @testset "Storage-origin re-export: destination in the storage's own country is never tariffed, regardless of cohort tags" begin
        product = Product("product")

        supplier_cn = Supplier("supplier_cn", Location(0.0, 0.0; country="CN"))
        add_product!(supplier_cn, product; unit_cost=8.0)
        supplier_de = Supplier("supplier_de", Location(0.0, 0.0; country="DE"))
        add_product!(supplier_de, product; unit_cost=12.0)
        storage = Storage("storage", Location(0.0, 0.0; country="US"))
        customer = Customer("customer", Location(0.0, 0.0; country="US"))

        l_cn = Lane(supplier_cn, storage; unit_cost=0)
        l_de = Lane(supplier_de, storage; unit_cost=0)
        l_out = Lane(storage, customer; unit_cost=0)

        network = SupplyChain(1)
        add_supplier!(network, supplier_cn)
        add_supplier!(network, supplier_de)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l_cn)
        add_lane!(network, l_de)
        add_lane!(network, l_out)
        add_demand!(network, customer, product, [150.0]; sales_price=1.0, lost_sales_cost=1.0)
        # CN->US and DE->US tariffs registered (goods already paid these
        # entering the storage's own country) - a purely domestic delivery
        # out of that same storage must not be charged again.
        add_tariff!(network, Tariff("CN", "US", 0.15))
        add_tariff!(network, Tariff("DE", "US", 0.10))

        policies = Dict((l_cn, product) => QuantityOrderingPolicy([100]),
                         (l_de, product) => QuantityOrderingPolicy([50]))
        final_state = simulate(network, policies)

        # The import legs (CN->US, DE->US) are genuine cross-border shipments
        # and are correctly taxed; what this test actually checks is that the
        # domestic re-export (storage(US) -> customer(US)) adds nothing on
        # top of that - not that the whole run is tariff-free.
        import_only = 0.15 * 8.0 * 100 + 0.10 * 12.0 * 50
        @test get_total_tariff_costs(final_state) == import_only
        @test final_state.metrics.tariff_costs == import_only
    end

    @testset "product-specific Tariff overrides the wildcard for that product only" begin
        productA = Product("A")
        productB = Product("B")

        supplier = Supplier("supplier", Location(0.0, 0.0; country="CN"))
        add_product!(supplier, productA; unit_cost=10.0)
        add_product!(supplier, productB; unit_cost=10.0)
        storage = Storage("storage", Location(0.0, 0.0; country="US"))
        customer = Customer("customer")

        # Distinct ids: Lane's equality/hash (SupplyChainModeling.jl) is
        # content-based (origin/destinations/times) when no id is given, so
        # two structurally-identical lanes without one collide into a single
        # Lane key everywhere a Lane is used as a Dict/Set key (lane_policies,
        # supply_chain.lanes_out, ...) - silently merging their per-lane
        # policies and duplicating every order/trip built from them.
        lA = Lane(supplier, storage; id="lane-A", unit_cost=0)
        lB = Lane(supplier, storage; id="lane-B", unit_cost=0)
        l_outA = Lane(storage, customer; unit_cost=0)

        network = SupplyChain(1)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, productA)
        add_product!(network, productB)
        add_lane!(network, lA)
        add_lane!(network, lB)
        add_lane!(network, l_outA)
        add_demand!(network, customer, productA, [0.0]; sales_price=1.0, lost_sales_cost=1.0)
        add_demand!(network, customer, productB, [0.0]; sales_price=1.0, lost_sales_cost=1.0)
        add_tariff!(network, Tariff("CN", "US", 0.20))               # wildcard
        add_tariff!(network, Tariff("CN", "US", 0.05; product=productB))  # override for B

        policies = Dict((lA, productA) => QuantityOrderingPolicy([100]),
                         (lB, productB) => QuantityOrderingPolicy([100]))
        final_state = simulate(network, policies)

        @test get_total_tariff_costs(final_state) ≈ 0.20 * 10.0 * 100 + 0.05 * 10.0 * 100
    end

    @testset "SimMetrics vs history-scan equivalence, with tariffs, across horizons and policies" begin
        Random.seed!(101)
        for horizon in [1, 2, 8]
            product = Product("product")

            supplier = Supplier("supplier", Location(0.0, 0.0; country="CN"))
            add_product!(supplier, product; unit_cost=6.0)
            storage = Storage("storage", Location(0.0, 0.0; country="US"))
            add_product!(storage, product; unit_holding_cost=0.2)
            customer = Customer("customer", Location(0.0, 0.0; country="US"))

            l1 = Lane(storage, customer; unit_cost=0.1)
            l2 = Lane(supplier, storage; unit_cost=0.05, time=1)

            network = SupplyChain(horizon)
            add_supplier!(network, supplier)
            add_storage!(network, storage)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l1)
            add_lane!(network, l2)
            add_demand!(network, customer, product, rand(4:12, horizon) * 1.0; sales_price=2.0, lost_sales_cost=1.0)
            add_tariff!(network, Tariff("CN", "US", 0.12))

            for policy in [NetSSOrderingPolicy(5, 20), OnHandUptoOrderingPolicy(15)]
                policies = Dict((l2, product) => policy)
                final_state = simulate(network, policies)
                assert_metrics_match_history(final_state)
                @test final_state.metrics.tariff_costs >= 0.0
            end
        end
    end
end
