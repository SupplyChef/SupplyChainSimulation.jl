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

    add_demand!(network, customer, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)

    policies = Dict((l2, product) => policy2)

    optimize!(policies, network)

    println(policy2)

    final_state = simulate(network, policies)

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

    add_demand!(network, customer, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)

    policies = Dict((l2, product) => policy2)

    optimize!(policies, network)

    println(policy2)

    final_state = simulate(network, policies)

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

    n() = begin
        network = SupplyChain(horizon)
        
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        demand = Poisson(10)
        add_demand!(network, customer, product, rand(demand, horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)

        return network
    end

    initial_states = [n() for i in 1:30]

    policies = Dict((l2, product) => policy2)
    
    optimize!(policies, initial_states...)

    println(policy2)

    final_state = simulate(initial_states[1], policies)

    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("demand: $(get_total_demand(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")
    true
end