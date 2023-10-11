@test begin #sS policy
    p = Single("product")

    customer = Customer("c")
    storage = Storage("s", Dict(p => 0.1))
    storage2 = Storage("s2")

    horizon = 20
    
    l = Lane(; origin = storage, destination = customer)
    l2 = Lane(; origin = storage2, destination = storage)

    policy2 = NetSSOrderingPolicy(0, 0)

    network = Network([], [storage, storage2], [customer], get_trips([l, l2], horizon), [p])

    initial_state = State(; on_hand_inventory = Dict(storage => Dict(p => 0), 
                                                     storage2 => Dict(p => 20 * horizon)), 
                            demand = Dict((customer, p) => repeat([10], horizon)),
                            policies = Dict((l2, p) => policy2))

    optimize!(network, horizon, initial_state)

    println(policy2)

    final_state = simulate(network, horizon, initial_state)

    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("demand: $(get_total_demand(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")
    
    get_total_lost_sales(final_state) == 0 &&  get_total_sales(final_state) == 10 * horizon && get_total_demand(final_state) == 10 * horizon
end

@test begin #sS policy
    p = Single("product")

    customer = Customer("c")
    storage = Storage("s", Dict(p => 0.1))
    storage2 = Storage("s2")

    horizon = 20
    
    l = Lane(; origin = storage, destination = customer, unit_cost = 0)
    l2 = Lane(; origin = storage2, destination = storage, unit_cost = 0, lead_time = 2)

    policy2 = NetSSOrderingPolicy(0, 0)

    network = Network([], [storage, storage2], [customer], get_trips([l, l2], horizon), [p])

    initial_state = State(; on_hand_inventory = Dict(storage => Dict(p => 0), 
                                                     storage2 => Dict(p => 20 * horizon)),
                            demand = Dict((customer, p) => repeat([10], horizon)),
                            policies = Dict((l2, p) => policy2))

    optimize!(network, horizon, initial_state)

    println(policy2)

    final_state = simulate(network, horizon, initial_state)

    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("demand: $(get_total_demand(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")

    get_total_lost_sales(final_state) == 20 &&  get_total_sales(final_state) == 10 * horizon - 20 && get_total_demand(final_state) == 10 * horizon
end

@test begin #sS policy
    p = Single("product")

    customer = Customer("c")
    storage = Storage("s", Dict(p => 0.1))
    storage2 = Storage("s2")

    horizon = 20
    
    l = Lane(; origin = storage, destination = customer, unit_cost = 0)
    l2 = Lane(; origin = storage2, destination = storage, unit_cost = 0, lead_time = 2)

    policy2 = NetSSOrderingPolicy(0, 0)

    network = Network([], [storage, storage2], [customer], get_trips([l, l2], horizon), [p])

    demand = Poisson(10)

    initial_states = [State(; on_hand_inventory = Dict(storage => Dict(p => 0), 
                                                     storage2 => Dict(p => 20 * horizon)), 
                            demand = Dict((customer, p) => rand(demand, horizon)),
                            policies = Dict((l2, p) => policy2)) for i in 1:30]

    optimize!(network, horizon, initial_states...)

    println(policy2)

    final_state = simulate(network, horizon, initial_states[1])

    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("demand: $(get_total_demand(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")
    true
end