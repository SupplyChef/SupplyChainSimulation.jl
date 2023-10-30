@testset "Docs" begin
    @test begin
        horizon = 20
  
        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=1.0)
        customer = Customer("customer")

        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage)

        network = SupplyChain(horizon)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)

        policy = OnHandUptoOrderingPolicy(0)
        policies = Dict((l2, product) => policy)

        initial_states = [State(; demand = Dict((customer, product) => rand(Poisson(10), horizon))) for i in 1:10]

        optimize!(network, policies, initial_states...)
        final_states = [simulate(network, policies, initial_state) for initial_state in initial_states]
        true
    end

    @test begin
        horizon = 50

        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)

        customer = Customer("customer")
        
        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage, fixed_cost=10)

        network = SupplyChain(horizon)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)

        policy = NetSSOrderingPolicy(0, 0)
        policies = Dict((l2, product) => policy)

        initial_states = [State(; demand = Dict((customer, product) => repeat([10], horizon))) for i in 1:1]

        optimize!(network, policies, initial_states...)

        println(policy)
        true
    end

    @test begin
        horizon = 50

        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)
        customer = Customer("customer")

        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage; fixed_cost=10, time=2)

        network = SupplyChain(horizon)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)

        policy = NetSSOrderingPolicy(0, 0)
        policies = Dict((l2, product) => policy)

        initial_states = [State(; demand = Dict((customer, product) => rand(Poisson(10), horizon))) for i in 1:20]

        optimize!(network, policies, initial_states...)

        println(policy)
        true
    end

    @test begin
        product = Product("product")

        customer = Customer("customer")
        retailer = Storage("retailer")
        add_product!(retailer, product; initial_inventory=20, unit_holding_cost=0.1)
        wholesaler = Storage("wholesaler")
        add_product!(wholesaler, product; initial_inventory=20, unit_holding_cost=0.1)
        factory = Storage("factory")
        add_product!(factory, product; initial_inventory=20, unit_holding_cost=0.1)
        supplier = Supplier("supplier")

        horizon = 20

        l = Lane(retailer, customer; unit_cost=0)
        l2 = Lane(wholesaler, retailer; unit_cost=0, time=2)
        l3 = Lane(factory, wholesaler; unit_cost=0, time=2)
        l4 = Lane(supplier, factory; unit_cost=0, time=4)

        policy2 = NetUptoOrderingPolicy(0)
        policy3 = NetUptoOrderingPolicy(0)
        policy4 = NetUptoOrderingPolicy(0)
        policies = Dict(
                        (l2, product) => policy2,
                        (l3, product) => policy3,
                        (l4, product) => policy4)

        network = SupplyChain(horizon)
        add_supplier!(network, supplier)
        add_storage!(network, factory)
        add_storage!(network, wholesaler)
        add_storage!(network, retailer)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)
        add_lane!(network, l3)
        add_lane!(network, l4)

        initial_states = [State(; demand = Dict((customer, product) => rand(Poisson(10), horizon))) for i in 1:30]

        optimize!(network, policies, initial_states...)
        true
    end
end