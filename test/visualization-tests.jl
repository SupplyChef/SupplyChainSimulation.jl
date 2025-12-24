using Distributions: rand, Poisson

@testset "Policies" begin
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

        plot_inventory_movement(final_states[1], product)

        true
    end
end