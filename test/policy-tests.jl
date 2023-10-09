using Distributions

@testset "Policies" begin
    @test begin
        horizon = 20

        product = Single("product")

        supplier = Supplier("supplier")
        storage = Storage("storage", Dict(product => 1.0))
        customer = Customer("customer")
        
        l1 = Lane(; origin = storage, destination = customer)
        l2 = Lane(; origin = supplier, destination = storage)

        network = Network([supplier], [storage], [customer], get_trips([l1, l2], horizon), [product])

        policy = OnHandUptoOrderingPolicy(0)

        initial_states = [State(; on_hand_inventory = Dict(storage => Dict(product => 0)), 
                                demand = Dict((customer, product) => rand(Poisson(10), horizon)),
                                policies = Dict((l2, product) => policy)) for i in 1:10]

        optimize!(network, horizon, initial_states...)

        println(policy)

        final_states = [simulate(network, horizon, initial_state) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")
        true
    end

    @test begin
        p = Single("product")

        customer = Customer("c")
        storage = Storage("s", Dict(p => 1.0))
        storage2 = Storage("s2")

        horizon = 20
        
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)
        l2 = Route(; origin = storage2, destinations = [storage], unit_cost = 0)

        policy = OnHandUptoOrderingPolicy(0)
        policy2 = OnHandUptoOrderingPolicy(0)

        network = Network([], [storage, storage2], [customer], get_trips([l, l2], horizon), [p])

        initial_state = State(; on_hand_inventory = Dict(storage => Dict(p => 0), 
                                                         storage2 => Dict(p => 20 * horizon)), 
                                in_transit_inventory = Dict(storage => Dict(p => repeat([0], horizon)), 
                                                            storage2 => Dict(p => repeat([0], horizon)), 
                                                            customer => Dict(p => repeat([0], horizon))), 
                                pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                                demand = Dict((customer, p) => repeat([10], horizon)),
                                policies = Dict((l, p) => policy, (l2, p) => policy2))

        optimize!(network, horizon, initial_state)

        println(policy2)

        final_state = simulate(network, horizon, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        println("holding costs: $(get_total_holding_costs(final_state))")

        set_parameter!(policy, [0.0])
        set_parameter!(policy2, [20.0])
        println(policy2)
        
        final_state = simulate(network, horizon, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        println("holding costs: $(get_total_holding_costs(final_state))")
        true
    end

    @test begin
        customer = Customer("c")
        storage = Storage("s")
        storage2 = Storage("s2")

        horizon = 20
        
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)
        l2 = Lane(; origin = storage2, destination = storage, unit_cost = 0)

        p = Single("product")

        policy2 = NetUptoOrderingPolicy(0)

        network = Network([], [storage, storage2], [customer], get_trips([l, l2], horizon), [p])

        initial_state = State(; on_hand_inventory = Dict(storage => Dict(p => 0), 
                                                         storage2 => Dict(p => 20 * horizon)), 
                                in_transit_inventory = Dict(storage => Dict(p => repeat([0], horizon)), 
                                                            storage2 => Dict(p => repeat([0], horizon)), 
                                                            customer => Dict(p => repeat([0], horizon))), 
                                pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                                demand = Dict((customer, p) => repeat([10], horizon)),
                                policies = Dict((l2, p) => policy2))

        optimize!(network, horizon, initial_state)

        println(policy2)

        final_state = simulate(network, horizon, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")

        set_parameter!(policy2, [20.0])
        println(policy2)
        
        final_state = simulate(network, horizon, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        true
    end

    @test begin
        customer1 = Customer("c1")
        customer2 = Customer("c2")
        storage = Storage("s")
        storage2 = Storage("s2")

        horizon = 20
        
        l11 = Lane(; origin = storage, destination = customer1, unit_cost = 0)
        l12 = Lane(; origin = storage, destination = customer2, unit_cost = 0)
        l2 = Lane(; origin = storage2, destination = storage, unit_cost = 0)

        p = Single("product")

        policy2 = NetUptoOrderingPolicy(0)

        network = Network([], [storage, storage2], [customer1, customer2], get_trips([l11, l12, l2], horizon), [p])

        initial_state = State(; on_hand_inventory = Dict(storage => Dict(p => 0), 
                                                         storage2 => Dict(p => 20 * horizon)), 
                                in_transit_inventory = Dict(storage => Dict(p => repeat([0], horizon)), 
                                                            storage2 => Dict(p => repeat([0], horizon)), 
                                                            customer1 => Dict(p => repeat([0], horizon)), 
                                                            customer2 => Dict(p => repeat([0], horizon))), 
                                pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                                demand = Dict((customer1, p) => repeat([10], horizon), (customer2, p) => repeat([10], horizon)),
                                policies = Dict((l2, p) => policy2))

        optimize!(network, horizon, initial_state)

        println("Optimized policy: $policy2")

        final_state = simulate(network, horizon, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")

        set_parameter!(policy2, [20.0])
        println(policy2)
        
        final_state = simulate(network, horizon, initial_state)

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
        
        l11 = Lane(; origin = storage1, destination = customer1, unit_cost = 0)
        l22 = Lane(; origin = storage2, destination = customer2, unit_cost = 0)
        l1 = Lane(; origin = storage, destination = storage1, unit_cost = 0)
        l2 = Lane(; origin = storage, destination = storage2, unit_cost = 0)
        l = Lane(; origin = supplier, destination = storage, unit_cost = 0)

        p = Single("product")

        policy = NetUptoOrderingPolicy(0)
        policy1 = NetUptoOrderingPolicy(0)
        policy2 = NetUptoOrderingPolicy(0)

        network = Network([supplier], [storage1, storage2, storage], [customer1, customer2], get_trips([l11, l22, l1, l2, l], horizon), [p])

        initial_state = State(; on_hand_inventory = Dict(
                                                        storage1 => Dict(p => 0), 
                                                        storage2 => Dict(p => 0), 
                                                        storage => Dict(p => 0)), 
                                demand = Dict(
                                            (customer1, p) => repeat([10], horizon), 
                                            (customer2, p) => repeat([10], horizon)),
                                policies = Dict(
                                                (l, p) => policy,
                                                (l1, p) => policy1,
                                                (l2, p) => policy2)
                        )

        optimize!(network, horizon, initial_state)

        println(policy)
        println(policy1)
        println(policy2)

        final_state = simulate(network, horizon, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")

        true
    end

    @test begin #quantity policy
        p = Single("product")
    
        customer = Customer("c")
        storage = Storage("s", Dict(p => 1.0))
        storage2 = Storage("s2")
    
        horizon = 20
        
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)
        l2 = Lane(; origin = storage2, destination = storage, unit_cost = 0, lead_time = 2)
    
        policy2 = QuantityOrderingPolicy(zeros(horizon))
    
        network = Network([], [storage, storage2], [customer], get_trips([l, l2], horizon), [p])
    
        demand = Poisson(10)
    
        initial_states = [State(; on_hand_inventory = Dict(storage => Dict(p => 0), 
                                                         storage2 => Dict(p => 20 * horizon)), 
                                in_transit_inventory = Dict(storage => Dict(p => repeat([0], horizon)), 
                                                            storage2 => Dict(p => repeat([0], horizon)), 
                                                            customer => Dict(p => repeat([0], horizon))), 
                                pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                                demand = Dict((customer, p) => rand(demand, horizon)),
                                policies = Dict((l2, p) => policy2)) for i in 1:10]
    
        optimize!(network, horizon, initial_states...)
    
        println(policy2)
    
        final_state = simulate(network, horizon, initial_states[1])
    
        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        println("holding costs: $(get_total_holding_costs(final_state))")
        true
    end
end

@testset "Beer game" begin
    @test begin
        p = Single("product")
    
        customer = Customer("customer")
        retailer = Storage("retailer", Dict(p => 1.0))
        wholesaler = Storage("wholesaler", Dict(p => 1.0))
        factory = Storage("factory", Dict(p => 1.0))
        supplier = Supplier("supplier")
    
        horizon = 20
        
        l = Lane(; origin = retailer, destination = customer, unit_cost = 0)
        l2 = Lane(; origin = wholesaler, destination = retailer, unit_cost = 0, lead_time = 2)
        l3 = Lane(; origin = factory, destination = wholesaler, unit_cost = 0, lead_time = 2)
        l4 = Lane(; origin = supplier, destination = factory, unit_cost = 0, lead_time = 4)

        policy2 = NetUptoOrderingPolicy(0)
        policy3 = NetUptoOrderingPolicy(0)
        policy4 = NetUptoOrderingPolicy(0)

        network = Network([supplier], [retailer, wholesaler, factory], [customer], get_trips([l, l2, l3, l4], horizon), [p])

        initial_state = State(; on_hand_inventory = Dict(
                                                        retailer => Dict(p => 20), 
                                                        wholesaler => Dict(p => 20), 
                                                        factory => Dict(p => 20)), 
                                demand = Dict((customer, p) => repeat([10], horizon)),
                                policies = Dict(
                                                (l2, p) => policy2,
                                                (l3, p) => policy3,
                                                (l4, p) => policy4)
                        )

        optimize!(network, horizon, initial_state)

        println(policy2)
        println(policy3)
        println(policy4)

        final_state = simulate(network, horizon, initial_state)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        true
    end

    @test begin
        p = Single("product")
    
        customer = Customer("customer")
        retailer = Storage("retailer", Dict(p => 1.0))
        wholesaler = Storage("wholesaler", Dict(p => 1.0))
        factory = Storage("factory", Dict(p => 1.0))
        supplier = Supplier("supplier")
    
        horizon = 20
        
        l = Lane(; origin = retailer, destination = customer, unit_cost = 0)
        l2 = Lane(; origin = wholesaler, destination = retailer, unit_cost = 0, lead_time = 2)
        l3 = Lane(; origin = factory, destination = wholesaler, unit_cost = 0, lead_time = 2)
        l4 = Lane(; origin = supplier, destination = factory, unit_cost = 0, lead_time = 4)

        policy2 = NetUptoOrderingPolicy(0)
        policy3 = NetUptoOrderingPolicy(0)
        policy4 = NetUptoOrderingPolicy(0)

        network = Network([supplier], [retailer, wholesaler, factory], [customer], get_trips([l, l2, l3, l4], horizon), [p])

        initial_states = [State(; on_hand_inventory = Dict(
                                                        retailer => Dict(p => 20), 
                                                        wholesaler => Dict(p => 20), 
                                                        factory => Dict(p => 20)), 
                                demand = Dict((customer, p) => rand(Poisson(10), horizon)),
                                policies = Dict(
                                                (l2, p) => policy2,
                                                (l3, p) => policy3,
                                                (l4, p) => policy4)
                        ) for i in 1:30]

        optimize!(network, horizon, initial_states...)

        println(policy2)
        println(policy3)
        println(policy4)

        final_states = [simulate(network, horizon, initial_state) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        true
    end
end