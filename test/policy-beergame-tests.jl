using Random

function beer_game(;scenario_count=30, optimize=true)
    Random.seed!(3)
    
    product = SupplyChainModeling.Product("product")

    customer = Customer("customer")
    retailer = Storage("retailer")
    add_product!(retailer, product; unit_holding_cost=0.1, initial_inventory=20)
    wholesaler = Storage("wholesaler")
    add_product!(wholesaler, product; unit_holding_cost=0.1, initial_inventory=20)
    factory = Storage("factory")
    add_product!(factory, product; unit_holding_cost=0.1, initial_inventory=20)
    supplier = Supplier("supplier")

    horizon = 200
    
    l = Lane(retailer, customer; unit_cost=0)
    l2 = Lane(wholesaler, retailer; unit_cost=0, time=2)
    l3 = Lane(factory, wholesaler; unit_cost=0, time= 2)
    l4 = Lane(supplier, factory; unit_cost=0, time=4)

    policy2 = BackwardCoverageOrderingPolicy([0.0, 0.0])
    policy3 = BackwardCoverageOrderingPolicy([0.0, 0.0])
    policy4 = BackwardCoverageOrderingPolicy([0.0, 0.0])

    n() = begin
        network = SupplyChain(horizon)
        
        add_supplier!(network, supplier)
        add_storage!(network, retailer)
        add_storage!(network, wholesaler)
        add_storage!(network, factory)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)
        add_lane!(network, l3)
        add_lane!(network, l4)

        add_demand!(network, customer, product, rand(Poisson(10), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)

        return network
    end

    initial_states = [n() for i in 1:scenario_count]
    println(initial_states)

    policies = Dict((l2, product) => policy2,
                    (l3, product) => policy3,
                    (l4, product) => policy4)
    
    if optimize
        optimize!(policies, initial_states...)
    end

    println(policy2)
    println(policy3)
    println(policy4)

    final_states = [simulate(initial_state, policies) for initial_state in initial_states]
    return final_states
end

@testset "Beer game" begin
    @test begin
        Random.seed!(3)

        product = Product("product")
    
        customer = Customer("customer")
        retailer = Storage("retailer")
        add_product!(retailer, product; unit_holding_cost=0.1, initial_inventory=20)
        wholesaler = Storage("wholesaler")
        add_product!(wholesaler, product; unit_holding_cost=0.1, initial_inventory=20)
        factory = Storage("factory")
        add_product!(factory, product; unit_holding_cost=0.1, initial_inventory=20)
        supplier = Supplier("supplier")
    
        horizon = 20
        
        l = Lane(retailer, customer; unit_cost=0)
        l2 = Lane(wholesaler, retailer; unit_cost=0, time=2)
        l3 = Lane(factory, wholesaler; unit_cost=0, time= 2)
        l4 = Lane(supplier, factory; unit_cost=0, time=4)

        policy2 = NetUptoOrderingPolicy(0)
        policy3 = NetUptoOrderingPolicy(0)
        policy4 = NetUptoOrderingPolicy(0)

        network = SupplyChain(horizon)
        
        add_supplier!(network, supplier)
        add_storage!(network, retailer)
        add_storage!(network, wholesaler)
        add_storage!(network, factory)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)
        add_lane!(network, l3)
        add_lane!(network, l4)

        add_demand!(network, customer, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict(
                        (l2, product) => policy2,
                        (l3, product) => policy3,
                        (l4, product) => policy4)

        optimize!(policies, network)

        println(policy2)
        println(policy3)
        println(policy4)

        final_state = simulate(network, policies)

        println("lost sales: $(get_total_lost_sales(final_state))")
        println("sales: $(get_total_sales(final_state))")
        println("demand: $(get_total_demand(final_state))")
        true
    end

    @test begin
        Random.seed!(3)

        product = Product("product")
    
        customer = Customer("customer")
        retailer = Storage("retailer")
        add_product!(retailer, product; unit_holding_cost=0.1, initial_inventory=20)
        wholesaler = Storage("wholesaler")
        add_product!(wholesaler, product; unit_holding_cost=0.1, initial_inventory=20)
        factory = Storage("factory")
        add_product!(factory, product; unit_holding_cost=0.1, initial_inventory=20)
        supplier = Supplier("supplier")
    
        horizon = 20
        
        l = Lane(retailer, customer; unit_cost=0)
        l2 = Lane(wholesaler, retailer; unit_cost=0, time=2)
        l3 = Lane(factory, wholesaler; unit_cost=0, time= 2)
        l4 = Lane(supplier, factory; unit_cost=0, time=4)

        policy2 = NetUptoOrderingPolicy(0)
        policy3 = NetUptoOrderingPolicy(0)
        policy4 = NetUptoOrderingPolicy(0)

        n() = begin
            network = SupplyChain(horizon)
        
            add_supplier!(network, supplier)
            add_storage!(network, retailer)
            add_storage!(network, wholesaler)
            add_storage!(network, factory)
            add_customer!(network, customer)
            add_product!(network, product)
            add_lane!(network, l)
            add_lane!(network, l2)
            add_lane!(network, l3)
            add_lane!(network, l4)

            add_demand!(network, customer, product, rand(Poisson(10), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)

            return network
        end

        initial_states = [n() for i in 1:30]

        policies = Dict(
                        (l2,product) => policy2,
                        (l3, product) => policy3,
                        (l4, product) => policy4)

        optimize!(policies, initial_states...)

        println(policy2)
        println(policy3)
        println(policy4)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        true
    end

    @test begin
        final_states = beer_game()

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        true
    end
end