@testset "Constraint Enforcement" begin
    @test begin
        # Storage capacity: more arrives than maximum_units allows; the excess
        # should be capped (not lost) and delayed, and accrue an overflow cost
        # for every period it's stuck waiting.
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product; maximum_units=5, overflow_unit_cost=2.0)
        l = Lane(storage, customer; unit_cost=0)
        storage2 = Storage("s2")
        l2 = Lane(storage2, storage; unit_cost=0, initial_arrivals=Dict(product => [10, 0]))

        network = SupplyChain(2)

        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        add_demand!(network, customer, product, [0.0, 0.0]; sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict((l, product) => OnHandUptoOrderingPolicy(0))
        final_state = simulate(network, policies)

        # 10 units arrive at time 1 but only 5 fit; the other 5 overflow at
        # time 1, are retried at time 2 (still no room, since nothing is
        # consumed), and overflow again there: 5 + 5 = 10 unit-periods of
        # overflow at a cost of 2.0 each.
        get_on_hand_inventory(final_state, storage, product) == 5 &&
        get_total_overflow_costs(final_state) == 20.0
    end

    @test begin
        # Minimum order quantity: a policy that would order less than a
        # lane's minimum_quantity should round up to the minimum instead of
        # placing a smaller (or no) order.
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product)
        supplier = Supplier("supplier")

        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=0, minimum_quantity=20)

        network = SupplyChain(1)

        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        add_demand!(network, customer, product, [0.0]; sales_price=1.0, lost_sales_cost=1.0)

        # on-hand starts at 0, so this policy would normally order 5 - below
        # the lane's minimum_quantity of 20.
        policies = Dict((l2, product) => OnHandUptoOrderingPolicy(5))
        final_state = simulate(network, policies)

        placed = collect(Base.Iterators.flatten(final_state.historical_orders))
        length(placed) == 1 && placed[1].quantity == 20
    end

    @test begin
        # A zero order should stay zero - MOQ rounding only applies to
        # strictly positive orders.
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product; initial_inventory=100)
        supplier = Supplier("supplier")

        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=0, minimum_quantity=20)

        network = SupplyChain(1)

        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        add_demand!(network, customer, product, [0.0]; sales_price=1.0, lost_sales_cost=1.0)

        # on-hand starts at 100, well above the upto=5 target, so the policy
        # orders 0.
        policies = Dict((l2, product) => OnHandUptoOrderingPolicy(5))
        final_state = simulate(network, policies)

        placed = collect(Base.Iterators.flatten(final_state.historical_orders))
        length(placed) == 0
    end
end
