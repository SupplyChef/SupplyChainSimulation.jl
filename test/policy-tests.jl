using Distributions

@testset "Policies" begin
    @test begin
        horizon = 20

        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)
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

        initial_states = [State(; 
                                demand = Dict((customer, product) => rand(Poisson(10), horizon)),
                                ) for i in 1:10]

        policies = Dict((l2, product) => policy)
        optimize!(network, policies, initial_states...)

        println(policy)

        final_states = [simulate(network, policies, initial_state) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")
        true
    end

    @test begin
        horizon = 20
        
        product = Product("product")

        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product; unit_holding_cost=0.1)
        storage2 = Storage("s2")
        add_product!(storage2, product; initial_inventory=20 * horizon)
        
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0)

        policy = OnHandUptoOrderingPolicy(0)
        policy2 = OnHandUptoOrderingPolicy(0)

        network = SupplyChain(horizon)
        
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        initial_state = State(; pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                                demand = Dict((customer, product) => repeat([10], horizon)))

        policies = Dict((l, product) => policy, (l2, product) => policy2)
        optimize!(network, policies, initial_state)

        println(policy2)

        final_state = simulate(network, policies, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        println("holding costs: $(get_total_holding_costs(final_state))")

        set_parameters!(policy, [0.0])
        set_parameters!(policy2, [20.0])
        println(policy2)
        
        final_state = simulate(network, policies, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        println("holding costs: $(get_total_holding_costs(final_state))")
        true
    end

    @test begin
        horizon = 20

        product = Product("product")

        customer = Customer("c")
        storage = Storage("s")
        storage2 = Storage("s2")
        add_product!(storage2, product; initial_inventory=20 * horizon)
    
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0)

        policy2 = NetUptoOrderingPolicy(0)

        network = SupplyChain(horizon)

        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        initial_state = State(; pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                                demand = Dict((customer, product) => repeat([10], horizon)))

        policies = Dict((l2, product) => policy2)

        optimize!(network, policies, initial_state)

        println(policy2)

        final_state = simulate(network, policies, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")

        set_parameters!(policy2, [20.0])
        println(policy2)
        
        final_state = simulate(network, policies, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        true
    end

    @test begin
        horizon = 20

        product = Product("product")

        customer1 = Customer("c1")
        customer2 = Customer("c2")
        storage = Storage("s")
        storage2 = Storage("s2")
        add_product!(storage2, product; initial_inventory=20 * horizon)
        
        l11 = Lane(storage, customer1; unit_cost=0)
        l12 = Lane(storage, customer2; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0)

        policy2 = NetUptoOrderingPolicy(0)

        network = SupplyChain(horizon)
        
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer1)
        add_customer!(network, customer2)
        add_product!(network, product)
        add_lane!(network, l11)
        add_lane!(network, l12)
        add_lane!(network, l2)

        initial_state = State(; pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                                demand = Dict((customer1, product) => repeat([10], horizon), (customer2, product) => repeat([10], horizon)))

        policies = Dict((l2, product) => policy2)
                                
        optimize!(network, policies, initial_state)

        println("Optimized policy: $policy2")

        final_state = simulate(network, policies, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")

        set_parameters!(policy2, [20.0])
        println(policy2)
        
        final_state = simulate(network, policies, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        true
    end

    @test begin
        customer1 = Customer("c1")
        customer2 = Customer("c2")
        storage1 = Storage("s1")
        storage2 = Storage("s2")
        storage = Storage("s")
        supplier = Supplier("supp")

        horizon = 20
        
        l11 = Lane(storage1, customer1; unit_cost=0)
        l22 = Lane(storage2, customer2; unit_cost=0)
        l1 = Lane(storage, storage1; unit_cost=0)
        l2 = Lane(storage, storage2; unit_cost=0)
        l = Lane(supplier, storage; unit_cost=0)

        product = Product("product")

        policy = NetUptoOrderingPolicy(0)
        policy1 = NetUptoOrderingPolicy(0)
        policy2 = NetUptoOrderingPolicy(0)

        network = SupplyChain(horizon)
        
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_storage!(network, storage1)
        add_storage!(network, storage2)
        add_customer!(network, customer1)
        add_customer!(network, customer2)
        add_product!(network, product)
        add_lane!(network, l11)
        add_lane!(network, l22)
        add_lane!(network, l1)
        add_lane!(network, l2)
        add_lane!(network, l)


        initial_state = State(; 
                                demand = Dict(
                                            (customer1, product) => repeat([10], horizon), 
                                            (customer2, product) => repeat([10], horizon)),
                        )

        policies = Dict(
                      (l, product) => policy,
                      (l1, product) => policy1,
                      (l2, product) => policy2)

        optimize!(network, policies, initial_state)

        println(policy)
        println(policy1)
        println(policy2)

        final_state = simulate(network, policies, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")

        true
    end

    @test begin #quantity policy
        horizon = 20
        
        product = Product("product")
    
        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product; unit_holding_cost=0.1)
        storage2 = Storage("s2")
        add_product!(storage2, product; initial_inventory=20 * horizon)
        
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0, time=2)
    
        policy2 = QuantityOrderingPolicy(zeros(horizon))
    
        network = SupplyChain(horizon)

        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)
    
        demand = Poisson(10)
    
        initial_states = [State(; pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                                  demand = Dict((customer, product) => rand(demand, horizon))) for i in 1:10]
    
        policies = Dict((l2, product) => policy2)
        optimize!(network, policies, initial_states...)
    
        println(policy2)
    
        final_state = simulate(network, policies, initial_states[1])
    
        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        println("holding costs: $(get_total_holding_costs(final_state))")
        true
    end

    @test begin #eoq
        horizon = 50

        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)
        customer = Customer("customer")
        
        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage; fixed_cost=10)

        network = SupplyChain(horizon)

        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)

        policy = NetSSOrderingPolicy(0, 0)

        initial_states = [State(; 
                                demand = Dict((customer, product) => repeat([10], horizon))) for i in 1:1]

        policies = Dict((l2, product) => policy)
        
        optimize!(network, policies, initial_states...)

        println(policy)

        final_states = [simulate(network, policies, initial_state) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")
        true
    end

    @test begin #safety stock
        horizon = 50

        product = SupplyChainModeling.Product("product")

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

        initial_states = [State(; 
                                demand = Dict((customer, product) => rand(Poisson(10), horizon))) for i in 1:20]

        policies = Dict((l2, product) => policy)
        optimize!(network, policies, initial_states...)

        println(policy)

        final_states = [simulate(network, policies, initial_state) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")
        true
    end

end