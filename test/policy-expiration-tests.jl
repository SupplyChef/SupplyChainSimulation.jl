using Distributions

@testset "Expiration" begin
    @test begin
        horizon = 20
        Random.seed!(3)

        product = SupplyChainModeling.Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1, maximum_age=0)
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

        #println(policy)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")
        
        get_total_lost_sales(final_states[1]) == 0 && get_total_sales(final_states[1]) == 188 && get_total_holding_costs(final_states[1]) == 0
    end

    @test begin
        horizon = 20
        Random.seed!(3)

        product = SupplyChainModeling.Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)
        customer = Customer("customer")
        
        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage; id="lane1", can_ship=vcat([true], repeat([false], 19)))
        l3 = Lane(supplier, storage; id="lane2", can_ship=vcat(repeat([false], 10), [true], repeat([false], 9)))

        n(i) = begin
            network = SupplyChain(horizon)
            
            add_supplier!(network, supplier)
            add_storage!(network, storage)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l1)
            add_lane!(network, l2)
            add_lane!(network, l3)

            if i == 1
                add_demand!(network, customer, product, vcat(repeat([10], Int(horizon / 2)), repeat([100], Int(horizon / 2))) * 1.0; sales_price=1.0, lost_sales_cost=1.0)
            elseif i == 2
                add_demand!(network, customer, product, vcat(repeat([1], Int(horizon / 2)), repeat([10], Int(horizon / 2))) * 1.0; sales_price=1.0, lost_sales_cost=1.0)
            end

            return network
        end

        policy2 = OnHandUptoOrderingPolicy(0)
        policy3 = NetSSOrderingPolicy(0, 0)

        initial_states = [n(i) for i in 1:2]

        policies = Dict((l2, product) => policy2, (l3, product) => policy3)
        optimize!(policies, initial_states...)

        #println(policy)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")

        true
    end
end