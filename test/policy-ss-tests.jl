@test begin #sS policy
    horizon = 20
    
    product = Product("product")

    customer = Customer("c")
    storage = Storage("s")
    add_product!(storage, product; unit_holding_cost=0.1)
    storage2 = Storage("s2")
    add_product!(storage2, product; initial_inventory=20 * horizon)
    
    l = Lane(storage, customer)
    l2 = Lane(storage2, storage)

    policy2 = NetSSOrderingPolicy(0, 0)

    network = SupplyChain(horizon)
        
    add_storage!(network, storage)
    add_storage!(network, storage2)
    add_customer!(network, customer)
    add_product!(network, product)
    add_lane!(network, l)
    add_lane!(network, l2)

    initial_state = State(; demand = Dict((customer, product) => repeat([10], horizon)))

    policies = Dict((l2, product) => policy2)

    optimize!(network, policies, initial_state)

    println(policy2)

    final_state = simulate(network, policies, initial_state)

    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("demand: $(get_total_demand(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")
    
    get_total_lost_sales(final_state) == 0 &&  get_total_sales(final_state) == 10 * horizon && get_total_demand(final_state) == 10 * horizon
end

@test begin #sS policy
    horizon = 20
    
    product = Product("product")

    customer = Customer("c")
    storage = Storage("s")
    add_product!(storage, product; unit_holding_cost=0.1)
    storage2 = Storage("s2")
    add_product!(storage2, product; initial_inventory=20 * horizon)
    
    
    l = Lane(storage, customer; unit_cost=0)
    l2 = Lane(storage2, storage; unit_cost=0, time=2)

    policy2 = NetSSOrderingPolicy(0, 0)

    network = SupplyChain(horizon)
        
    add_storage!(network, storage)
    add_storage!(network, storage2)
    add_customer!(network, customer)
    add_product!(network, product)
    add_lane!(network, l)
    add_lane!(network, l2)

    initial_state = State(; demand = Dict((customer, product) => repeat([10], horizon)))

    policies = Dict((l2, product) => policy2)

    optimize!(network, policies, initial_state)

    println(policy2)

    final_state = simulate(network, policies, initial_state)

    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("demand: $(get_total_demand(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")

    get_total_lost_sales(final_state) == 20 &&  get_total_sales(final_state) == 10 * horizon - 20 && get_total_demand(final_state) == 10 * horizon
end

@test begin #sS policy
    horizon = 20
    
    product = Product("product")

    customer = Customer("c")
    storage = Storage("s")
    add_product!(storage, product; unit_holding_cost=0.1)
    storage2 = Storage("s2")
    add_product!(storage, product; initial_inventory=20 * horizon)
    
    l = Lane(storage, customer; unit_cost=0)
    l2 = Lane(storage2, storage; unit_cost=0, time=2)

    policy2 = NetSSOrderingPolicy(0, 0)

    network = SupplyChain(horizon)
        
    add_storage!(network, storage)
    add_storage!(network, storage2)
    add_customer!(network, customer)
    add_product!(network, product)
    add_lane!(network, l)
    add_lane!(network, l2)

    demand = Poisson(10)

    initial_states = [State(; demand = Dict((customer, product) => rand(demand, horizon))) for i in 1:30]

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