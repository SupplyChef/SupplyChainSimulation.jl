using Distributions: Poisson
using Random

@testset "Policies" begin
    @test begin
        horizon = 20
        Random.seed!(3)

        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)
        customer = Customer("customer")
        
        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage)

        n() = begin
            network = SupplyChain(horizon)
            
            add_supplier!(network, supplier)
            add_storage!(network, storage)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l1)
            add_lane!(network, l2)

            add_demand!(network, customer, product, rand(Poisson(10), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)

            return network
        end

        policy = OnHandUptoOrderingPolicy(0)

        initial_states = [n() for i in 1:10]

        policies = Dict((l2, product) => policy)
        optimize!(policies, initial_states...)

        println(policy)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")
        
        get_total_lost_sales(final_states[1]) == 0 && get_total_sales(final_states[1]) == 188
    end

    @test begin
        horizon = 20
        Random.seed!(3)
        
        product = SupplyChainModeling.Product("product")

        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product; unit_holding_cost=0.1)
        storage2 = Storage("s2")
        add_product!(storage2, product; initial_inventory=20 * horizon)
        
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0)

        # No policy is attached to (l, product): customer-facing lanes never
        # consult a policy - place_orders(::Customer, ...) derives quantity
        # purely from state.demand - so it would be inert.
        policy2 = OnHandUptoOrderingPolicy(0)

        network = SupplyChain(horizon)

        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        add_demand!(network, customer, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict((l2, product) => policy2)
        # No BackwardCoverageOrderingPolicy and no exact-value assertion tied
        # directly to optimize!'s raw output here, so the record_history=false
        # fast path is safe.
        optimize!(policies, network; cost_function=metrics_cost_function, record_history=false)

        println(policy2)

        final_state = simulate(network, policies)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        println("holding costs: $(get_total_holding_costs(final_state))")

        set_parameters!(policy2, [20.0])
        println(policy2)

        final_state = simulate(network, policies)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        println("holding costs: $(get_total_holding_costs(final_state))")

        # l2 has zero lead time and demand is a deterministic 10/period;
        # an on-hand-upto-20 policy should comfortably cover 10/period
        # demand every period with no lost sales.
        get_total_lost_sales(final_state) == 0 && get_total_sales(final_state) == 10 * horizon
    end

    @test begin
        horizon = 20
        Random.seed!(3)

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
        
        add_demand!(network, customer, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict((l2, product) => policy2)

        # No BackwardCoverageOrderingPolicy and no exact-value assertion tied
        # directly to optimize!'s raw output here, so the record_history=false
        # fast path is safe.
        optimize!(policies, network; cost_function=metrics_cost_function, record_history=false)

        println(policy2)

        final_state = simulate(network, policies)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")

        set_parameters!(policy2, [20.0])
        println(policy2)
        
        final_state = simulate(network, policies)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        true
    end

    @test begin
        horizon = 20
        Random.seed!(3)

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

        add_demand!(network, customer1, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)
        add_demand!(network, customer2, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict((l2, product) => policy2)
                                
        # No BackwardCoverageOrderingPolicy and no exact-value assertion tied
        # directly to optimize!'s raw output here, so the record_history=false
        # fast path is safe.
        optimize!(policies, network; cost_function=metrics_cost_function, record_history=false)

        println("Optimized policy: $policy2")

        final_state = simulate(network, policies)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")

        set_parameters!(policy2, [20.0])
        println(policy2)
        
        final_state = simulate(network, policies)

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
        Random.seed!(3)
        
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

        add_demand!(network, customer1, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)
        add_demand!(network, customer2, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict(
                      (l, product) => policy,
                      (l1, product) => policy1,
                      (l2, product) => policy2)

        # No BackwardCoverageOrderingPolicy and no exact-value assertion tied
        # directly to optimize!'s raw output here, so the record_history=false
        # fast path is safe.
        optimize!(policies, network; cost_function=metrics_cost_function, record_history=false)

        println(policy)
        println(policy1)
        println(policy2)

        final_state = simulate(network, policies)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")

        true
    end

    @test begin #quantity policy
        horizon = 20
        Random.seed!(3)
        
        product = Product("product")
    
        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product; unit_holding_cost=0.1)
        storage2 = Storage("s2")
        add_product!(storage2, product; initial_inventory=20 * horizon)
        
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0, time=2)
    
        policy2 = QuantityOrderingPolicy(zeros(horizon))
    
        n() = begin
            network = SupplyChain(horizon)

            add_storage!(network, storage)
            add_storage!(network, storage2)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l)
            add_lane!(network, l2)
    
            add_demand!(network, customer, product, rand(Poisson(10), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)

            return network
        end
    
        initial_states = [n() for i in 1:10]
    
        policies = Dict((l2, product) => policy2)
        # No BackwardCoverageOrderingPolicy and no exact-value assertion tied
        # directly to optimize!'s raw output here, so the record_history=false
        # fast path is safe.
        optimize!(policies, initial_states...; cost_function=metrics_cost_function, record_history=false)
    
        println(policy2)
    
        final_state = simulate(initial_states[1], policies)
    
        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        println("holding costs: $(get_total_holding_costs(final_state))")
        true
    end

    @test begin #eoq
        horizon = 50
        Random.seed!(3)

        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)
        customer = Customer("customer")
        
        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage; fixed_cost=10)

        n() = begin
            network = SupplyChain(horizon)

            add_supplier!(network, supplier)
            add_storage!(network, storage)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l1)
            add_lane!(network, l2)

            add_demand!(network, customer, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)

            return network
        end

        policy = NetSSOrderingPolicy(0, 0)

        initial_states = [n() for i in 1:1]

        policies = Dict((l2, product) => policy)
        
        # No BackwardCoverageOrderingPolicy and no exact-value assertion tied
        # directly to optimize!'s raw output here, so the record_history=false
        # fast path is safe.
        optimize!(policies, initial_states...; cost_function=metrics_cost_function, record_history=false)

        println(policy)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")
        true
    end

    @test begin #safety stock
        horizon = 50
        Random.seed!(3)

        product = SupplyChainModeling.Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)
        customer = Customer("customer")
        
        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage; fixed_cost=10, time=2)

        n() = begin
            network = SupplyChain(horizon)

            add_supplier!(network, supplier)
            add_storage!(network, storage)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l1)
            add_lane!(network, l2)

            add_demand!(network, customer, product, rand(Poisson(10), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)

            return network
        end

        policy = NetSSOrderingPolicy(0, 0)

        initial_states = [n() for i in 1:20]

        policies = Dict((l2, product) => policy)
        # No BackwardCoverageOrderingPolicy and no exact-value assertion tied
        # directly to optimize!'s raw output here, so the record_history=false
        # fast path is safe.
        optimize!(policies, initial_states...; cost_function=metrics_cost_function, record_history=false)

        println(policy)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")
        true
    end

    @test begin #product quantity policy and single order policy
        horizon = 20
        Random.seed!(3)

        product = Product("product")

        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product; unit_holding_cost=0.1)
        storage2 = Storage("s2")
        add_product!(storage2, product; initial_inventory=20 * horizon)

        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0, time=2)

        product_quantity_policy = ProductQuantityOrderingPolicy(5, 3)
        single_order_policy = SingleOrderOrderingPolicy(3, 5)

        n() = begin
            network = SupplyChain(horizon)

            add_storage!(network, storage)
            add_storage!(network, storage2)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l)
            add_lane!(network, l2)

            add_demand!(network, customer, product, rand(Poisson(10), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)

            return network
        end

        initial_state = n()

        for policy in [product_quantity_policy, single_order_policy]
            policies = Dict((l2, product) => policy)

            final_state = simulate(initial_state, policies)

            println("lost sales: $(get_total_lost_sales(final_state))")
            println("sales: $(get_total_sales(final_state))")
            println("demand: $(get_total_demand(final_state))")
            println("holding costs: $(get_total_holding_costs(final_state))")
        end
        true
    end

end