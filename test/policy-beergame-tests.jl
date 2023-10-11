@testset "Beer game" begin
    @test begin
        p = Single("product")
    
        customer = Customer("customer")
        retailer = Storage("retailer", Dict(p => 0.1))
        wholesaler = Storage("wholesaler", Dict(p => 0.1))
        factory = Storage("factory", Dict(p => 0.1))
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
        retailer = Storage("retailer", Dict(p => 0.1))
        wholesaler = Storage("wholesaler", Dict(p => 0.1))
        factory = Storage("factory", Dict(p => 0.1))
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

    @test begin
        p = Single("product")
    
        customer = Customer("customer")
        retailer = Storage("retailer", Dict(p => 0.1))
        wholesaler = Storage("wholesaler", Dict(p => 0.1))
        factory = Storage("factory", Dict(p => 0.1))
        supplier = Supplier("supplier")
    
        horizon = 200
        
        l = Lane(; origin = retailer, destination = customer, unit_cost = 0)
        l2 = Lane(; origin = wholesaler, destination = retailer, unit_cost = 0, lead_time = 2)
        l3 = Lane(; origin = factory, destination = wholesaler, unit_cost = 0, lead_time = 2)
        l4 = Lane(; origin = supplier, destination = factory, unit_cost = 0, lead_time = 4)

        policy2 = BackwardCoverageOrderingPolicy([0.0, 0.0])
        policy3 = BackwardCoverageOrderingPolicy([0.0, 0.0])
        policy4 = BackwardCoverageOrderingPolicy([0.0, 0.0])

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