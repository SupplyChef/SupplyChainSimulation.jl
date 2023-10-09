@test begin #cover policy
    p = Single("product")

    customer = Customer("c")
    storage = Storage("s", Dict(p => 1.0))
    storage2 = Storage("s2")

    horizon = 20
    
    l = Lane(; origin = storage, destination = customer, unit_cost = 0)
    l2 = Lane(; origin = storage2, destination = storage, unit_cost = 0, lead_time = 2)

    policy = OnHandUptoOrderingPolicy(0)
    policy2 = ForwardCoverageOrderingPolicy(0)

    network = Network([], [storage, storage2], [customer], get_trips([l, l2], horizon), [p])

    demand = Poisson(10)

    initial_states = [State(; on_hand_inventory = Dict(storage => Dict(p => 0), 
                                                     storage2 => Dict(p => 20 * horizon)), 
                            in_transit_inventory = Dict(storage => Dict(p => repeat([0], horizon)), 
                                                        storage2 => Dict(p => repeat([0], horizon)), 
                                                        customer => Dict(p => repeat([0], horizon))), 
                            pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                            demand = Dict((customer, p) => rand(demand, horizon)),
                            policies = Dict((l, p) => policy, (l2, p) => policy2)) for i in 1:10]

    optimize!(network, horizon, initial_states...)

    println(policy)
    println(policy2)

    final_state = simulate(network, horizon, initial_states[1])

    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("demand: $(get_total_demand(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")
    true
end

@test begin #cover policy
    store_count = 100

    p = Single("product")

    customers = [Customer("c$i") for i in 1:store_count]
    storage = Storage("s", Dict(p => 1.0))
    storage2 = Storage("s2")

    horizon = 20
    
    lanes = [Lane(; origin = storage, destination = customers[i], unit_cost = 0) for i in 1:store_count]
    l0 = Lane(; origin = storage2, destination = storage, unit_cost = 0, lead_time = 2)

    policy = OnHandUptoOrderingPolicy(0)
    policy2 = ForwardCoverageOrderingPolicy(0)
    policies = Dict((l0, p) => policy2)
    println(policies)

    network = Network([], [storage, storage2], customers, get_trips(vcat(lanes, [l0]), horizon), [p])

    demand = Poisson(10)

    initial_states = [State(; on_hand_inventory = Dict(storage => Dict(p => 0), 
                                                     storage2 => Dict(p => 200000 * horizon)), 
                            in_transit_inventory = Dict(vcat([storage => Dict(p => repeat([0], horizon)), 
                                                        storage2 => Dict(p => repeat([0], horizon))], 
                                                        [customers[i] => Dict(p => repeat([0], horizon)) for i in 1:store_count])), 
                            pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                            demand = Dict([(customers[i], p) => rand(demand, horizon) for i in 1:store_count]...),
                            policies = policies) for j in 1:10]

    #println(network)
    optimize!(network, horizon, initial_states...; cost_function=s->get_total_lost_sales(s) + 0.00001 * get_total_orders(s))

    println(policy)
    println(policy2)

    final_state = simulate(network, horizon, initial_states[1])

    println("demand: $(get_total_demand(final_state))")
    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")
    true
end