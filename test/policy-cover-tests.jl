@test begin #cover policy
    horizon = 20

    product = Product("product")

    customer = Customer("c")
    storage = Storage("s")
    add_product!(storage, product; unit_holding_cost=0.1)
    storage2 = Storage("s2")
    add_product!(storage2, product; initial_inventory=20 * horizon)
    
    l = Lane(storage, customer; unit_cost=0)
    l2 = Lane(storage2, storage; unit_cost=0, time=2)

    policy = OnHandUptoOrderingPolicy(0)
    policy2 = ForwardCoverageOrderingPolicy(0)

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

    policies = Dict((l, product) => policy, (l2, product) => policy2)
                            
    optimize!(network, policies, initial_states...)

    println(policy)
    println(policy2)

    final_state = simulate(network, policies, initial_states[1])

    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("demand: $(get_total_demand(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")
    true
end

@test begin #cover policy
    store_count = 100
    horizon = 20

    product = Product("product")

    customers = [Customer("c$i") for i in 1:store_count]
    storage = Storage("s")
    add_product!(storage, product; unit_holding_cost=0.1)
    storage2 = Storage("s2")
    add_product!(storage2, product; initial_inventory=200000 * horizon)
    
    lanes = [Lane(storage, customers[i]; unit_cost=0) for i in 1:store_count]
    l0 = Lane(storage2, storage; unit_cost=0, time=2)

    policy = OnHandUptoOrderingPolicy(0)
    policy2 = ForwardCoverageOrderingPolicy(0)
    policies = Dict((l0, product) => policy2)

    network = SupplyChain(horizon)
        
    add_storage!(network, storage)
    add_storage!(network, storage2)
    for customer in customers
        add_customer!(network, customer)
    end
    add_product!(network, product)
    for l in lanes
        add_lane!(network, l)
    end
    add_lane!(network, l0)

    demand = Poisson(10)

    initial_states = [State(; pending_outbound_order_lines = Dict(storage => Set{OrderLine}(), storage2 => Set{OrderLine}()),
                              demand = Dict([(customers[i], product) => rand(demand, horizon) for i in 1:store_count]...)) for j in 1:10]

    #println(network)
    optimize!(network, policies, initial_states...; cost_function=s->get_total_lost_sales(s) + 0.00001 * get_total_orders(s))

    println(policy)
    println(policy2)

    final_state = simulate(network, policies, initial_states[1])

    println("demand: $(get_total_demand(final_state))")
    println("lost sales: $(get_total_lost_sales(final_state))")
    println("sales: $(get_total_sales(final_state))")
    println("holding costs: $(get_total_holding_costs(final_state))")
    true
end