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

        # (l, product)'s policy is inert here: place_orders(::Customer, ...)
        # derives quantity purely from state.demand and never consults it.
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
    # Per-product MOQ and order multiple on a Supplier's add_product!.
    # `policy_order` is what the policy asks for (on-hand starts at 0).
    function supplier_order_quantity(policy_order; lane_moq=0, moq=0, multiple=1)
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product)
        supplier = Supplier("supplier")
        add_product!(supplier, product; unit_cost=0, minimum_order_quantity=moq, order_multiple=multiple)
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=0, minimum_quantity=lane_moq)
        network = SupplyChain(1)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)
        add_demand!(network, customer, product, [0.0]; sales_price=1.0, lost_sales_cost=1.0)
        final_state = simulate(network, Dict((l2, product) => OnHandUptoOrderingPolicy(policy_order)))
        placed = collect(Base.Iterators.flatten(final_state.historical_orders))
        isempty(placed) ? 0 : only(placed).quantity
    end

    @test supplier_order_quantity(5; moq=20) == 20                 # SKU MOQ lifts a small order
    @test supplier_order_quantity(30; moq=20) == 30                # above MOQ: untouched
    @test supplier_order_quantity(30; multiple=12) == 36           # case pack rounds up
    @test supplier_order_quantity(36; multiple=12) == 36           # already a multiple
    @test supplier_order_quantity(5; moq=25, multiple=12) == 36    # MOQ itself rounded up to a multiple
    @test supplier_order_quantity(5; lane_moq=20, moq=30) == 30    # larger of lane and SKU MOQ binds
    @test supplier_order_quantity(5; lane_moq=40, moq=30) == 40
    @test supplier_order_quantity(5; lane_moq=20, multiple=12) == 24
    @test supplier_order_quantity(0; moq=20, multiple=12) == 0     # zero stays zero
    # Lane minimum_quantity is a total across products; a shortfall is topped
    # up by the TopUpRule. `orders` maps each product's name to what its
    # policy asks for (on-hand starts at 0); returns name => quantity placed.
    function lane_order_quantities(orders; lane_moq, rule=ProportionalTopUp(), multiples=Dict{String,Int}())
        customer = Customer("c")
        storage = Storage("s")
        supplier = Supplier("supplier")
        network = SupplyChain(1)
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=0, minimum_quantity=lane_moq)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_lane!(network, l)
        add_lane!(network, l2)
        products = Dict(name => Product(name) for name in keys(orders))
        policies = Dict{Tuple{Lane, Product}, OnHandUptoOrderingPolicy}()
        for (name, product) in products
            add_product!(storage, product)
            add_product!(supplier, product; unit_cost=0, order_multiple=get(multiples, name, 1))
            add_product!(network, product)
            add_demand!(network, customer, product, [0.0]; sales_price=1.0, lost_sales_cost=1.0)
            policies[(l2, product)] = OnHandUptoOrderingPolicy(orders[name])
        end
        final_state = simulate(network, policies; top_up_rule=rule)
        placed = collect(Base.Iterators.flatten(final_state.historical_orders))
        Dict(o.product.name => o.quantity for o in placed)
    end

    # Already above the lane total: nothing is added (each product is not lifted separately).
    @test lane_order_quantities(Dict("a" => 60, "b" => 40); lane_moq=100) == Dict("a" => 60, "b" => 40)
    # 80 < 100: the 20 shortfall is split 3:1 by default.
    @test lane_order_quantities(Dict("a" => 60, "b" => 20); lane_moq=100) == Dict("a" => 75, "b" => 25)
    # Alternative rule: everything goes to the largest order.
    @test lane_order_quantities(Dict("a" => 60, "b" => 20); lane_moq=100, rule=LargestOrderTopUp()) == Dict("a" => 80, "b" => 20)
    # A product that ordered nothing is never created by the top-up.
    @test lane_order_quantities(Dict("a" => 60, "b" => 0); lane_moq=100) == Dict("a" => 100)
    # Top-ups are whole multiples of the product's case pack.
    @test lane_order_quantities(Dict("a" => 60, "b" => 20); lane_moq=100, rule=LargestOrderTopUp(), multiples=Dict("a" => 12)) == Dict("a" => 84, "b" => 20)
    # A custom rule plugs in through top_up!.
    struct SmallestOrderTopUp <: TopUpRule end
    SupplyChainSimulation.top_up!(::SmallestOrderTopUp, q::Vector{Int}, m::Vector{Int}, shortfall::Int) =
        (i = argmin(q); q[i] += cld(shortfall, m[i]) * m[i]; nothing)
    @test lane_order_quantities(Dict("a" => 60, "b" => 20); lane_moq=100, rule=SmallestOrderTopUp()) == Dict("a" => 60, "b" => 40)
end
