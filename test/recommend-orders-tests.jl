@testset "recommend_orders" begin
    # supplier -> storage -> customer, storage starts with `on_hand` units
    function build(; on_hand=5, horizon=4, minimum_quantity=0)
        product = Product("product")
        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; initial_inventory=on_hand)
        customer = Customer("customer")
        l1 = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=0, time=1, minimum_quantity=minimum_quantity)
        network = SupplyChain(horizon)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)
        add_demand!(network, customer, product, fill(3.0, horizon); sales_price=1.0, lost_sales_cost=1.0)
        return network, l2, product
    end

    @testset "orders up to from the seeded on-hand stock" begin
        network, l2, product = build(on_hand=5)
        @test recommend_orders(network, Dict((l2, product) => OnHandUptoOrderingPolicy(15))) == Dict((l2, product) => 10)

        network, l2, product = build(on_hand=12)
        @test recommend_orders(network, Dict((l2, product) => OnHandUptoOrderingPolicy(15))) == Dict((l2, product) => 3)
    end

    @testset "nothing to order -> no entry; customer demand is not an order" begin
        network, l2, product = build(on_hand=20)
        @test isempty(recommend_orders(network, Dict((l2, product) => OnHandUptoOrderingPolicy(15))))
    end

    @testset "minimum_quantity is applied" begin
        network, l2, product = build(on_hand=14, minimum_quantity=8)
        @test recommend_orders(network, Dict((l2, product) => OnHandUptoOrderingPolicy(15))) == Dict((l2, product) => 8)
    end

    @testset "time selects the period" begin
        network, l2, product = build()
        policies = Dict((l2, product) => QuantityOrderingPolicy([10, 20, 0, 40]))
        @test recommend_orders(network, policies; time=1) == Dict((l2, product) => 10)
        @test recommend_orders(network, policies; time=2) == Dict((l2, product) => 20)
        @test isempty(recommend_orders(network, policies; time=3))
        @test recommend_orders(network, policies; time=4) == Dict((l2, product) => 40)
        @test_throws ArgumentError recommend_orders(network, policies; time=0)
        @test_throws ArgumentError recommend_orders(network, policies; time=5)
    end

    @testset "matches simulate's period-1 orders (historical_orders) on a multi-echelon chain" begin
        product = Product("product")
        customer = Customer("customer")
        retailer = Storage("retailer")
        add_product!(retailer, product; initial_inventory=4)
        wholesaler = Storage("wholesaler")
        add_product!(wholesaler, product; initial_inventory=6)
        supplier = Supplier("supplier")
        l1 = Lane(retailer, customer; unit_cost=0)
        l2 = Lane(wholesaler, retailer; unit_cost=0, time=1)
        l3 = Lane(supplier, wholesaler; unit_cost=0, time=2)
        network = SupplyChain(10)
        add_supplier!(network, supplier)
        add_storage!(network, retailer)
        add_storage!(network, wholesaler)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)
        add_lane!(network, l3)
        add_demand!(network, customer, product, fill(5.0, 10); sales_price=1.0, lost_sales_cost=1.0)
        policies = Dict((l2, product) => NetSSOrderingPolicy(8, 20), (l3, product) => NetSSOrderingPolicy(10, 30))

        recommended = recommend_orders(network, policies)

        state = simulate(network, policies)
        expected = Dict{Tuple{Lane, Product}, Int64}()
        for o in Base.Iterators.flatten(state.historical_orders)
            (o.creation_time == 1 && !(o.destination isa Customer)) || continue
            expected[(o.trip.route, o.product)] = get(expected, (o.trip.route, o.product), 0) + o.quantity
        end
        @test !isempty(expected)
        @test recommended == expected
    end

    @testset "inputs are not modified" begin
        network, l2, product = build(on_hand=5)
        policy = OnHandUptoOrderingPolicy(15)
        recommend_orders(network, Dict((l2, product) => policy))
        @test policy.upto == 15
        @test get_initial_inventory(first(network.storages), product) == 5
    end

    @testset "history-reading policies are rejected" begin
        network, l2, product = build()
        @test_throws ArgumentError recommend_orders(network, Dict((l2, product) => BackwardCoverageOrderingPolicy([1.0, 1.0])))
    end
end
