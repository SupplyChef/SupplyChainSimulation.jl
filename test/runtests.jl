using SupplyChainModeling

using Distributions:Poisson
using Test

using SupplyChainSimulation

@testset "SupplyChain" begin
    @test begin
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(storage, customer; unit_cost=0)

        network = SupplyChain(1)

        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)

        get_sorted_locations(network) == [storage, customer]
    end

    @test begin
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(customer, storage; unit_cost=0)

        network = SupplyChain(1)
        
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)

        get_sorted_locations(network) == [customer, storage]
    end

    @test begin
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        storage2 = Storage("s2")
        
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0)

        network = SupplyChain(1)
        
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        get_sorted_locations(network) == [storage2, storage, customer]
    end

    @test begin
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        storage2 = Storage("s2")
        
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0)

        network = SupplyChain(1)
        
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        get_downstream_customers(network, storage) == [customer] && get_downstream_customers(network, storage2) == [customer]
    end
end

@testset "State" begin
    @test begin
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(storage, customer; unit_cost=0)
        storage2 = Storage("s2")
        l2 = Lane(storage2, storage; unit_cost=0, initial_arrivals=Dict(product => [10, 0]))

        network = SupplyChain(1)
        
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        initial_state = State(;demand = Dict((customer, product) => [0, 0]))

        get_inbound_orders(initial_state, storage, product, 1) == 0 && 
        get_outbound_orders(initial_state, storage, product, 1) == 0
        #get_net_inventory(initial_state, storage, product, 1) == 10
    end

    @test begin
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        storage2 = Storage("s2")
        add_product!(storage2, product; initial_inventory=100)
        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0, initial_arrivals=Dict(product => [10, 0]))

        o = OrderLine(0, l.origin, l.destinations[1], product, 5, 1)
        o2 = OrderLine(0, l2.origin, l2.destinations[1], product, 15, 1)

        network = SupplyChain(1)
        
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        initial_state = State(; pending_outbound_order_lines = Dict(storage => [o], storage2 => [o2]),
                                demand = Dict((customer, product) => [0, 0]))

        #println("inbound $(get_inbound_orders(initial_state, storage, product, 1))") 
        #println("outbound $(get_outbound_orders(initial_state, storage, product, 1))")
        #println("net $(get_net_inventory(initial_state, storage, product, 1))")
        
        @info get_inbound_orders(initial_state, storage, product, 1)
        @info get_outbound_orders(initial_state, storage, product, 1)
        
        get_inbound_orders(initial_state, storage, product, 1) == 15 && 
        get_outbound_orders(initial_state, storage, product, 1) == 5 
        #get_net_inventory(initial_state, storage, product, 1) == 20
    end

    @test begin
        product = Product("product")
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(storage, customer; unit_cost=0)
        storage2 = Storage("s2")
        l2 = Lane(storage2, storage; unit_cost=0, initial_arrivals=Dict(product => [10, 0]))

        network = SupplyChain(2)
        
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        initial_state = State(; demand = Dict((customer, product) => [0, 0]))

        policies = Dict((l, product) => OnHandUptoOrderingPolicy(0))
        final_state = simulate(network, policies, initial_state)

        final_state.on_hand_inventory[(storage, product)] == 10 && 
        length(collect(Base.Iterators.flatten(final_state.historical_orders))) == 0 && 
        length(collect(Base.Iterators.flatten(final_state.historical_filled_orders))) == 0
    end

    @test begin
        product = Product("product")

        customer = Customer("c")
        storage = Storage("s")
        add_product!(storage, product; initial_inventory=10)
        l = Lane(storage, customer; unit_cost=0)
        storage2 = Storage("s2")
        add_product!(storage2, product)
        l2 = Lane(storage2, storage; unit_cost=0, initial_arrivals=Dict(product => [10, 0]))

        network = SupplyChain(2)
        
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)

        initial_state = State(; demand = Dict((customer, product) => [10, 10]))

        policies = Dict((l, product) => OnHandUptoOrderingPolicy(0))
        final_state = simulate(network, policies, initial_state)

        println(get_total_holding_costs(final_state))

        @info final_state.on_hand_inventory[(storage, product)]
        @info length(collect(Base.Iterators.flatten(final_state.historical_orders)))
        @info length(collect(Base.Iterators.flatten(final_state.historical_filled_orders)))
        final_state.on_hand_inventory[(storage, product)] == 0 && 
        length(collect(Base.Iterators.flatten(final_state.historical_orders))) == 2 && 
        length(collect(Base.Iterators.flatten(final_state.historical_filled_orders))) == 2
    end

    @test begin
        demand_rate = 1000
        ordering_cost = 2 
        holding_cost_rate = 5
        eoq_quantity(demand_rate, ordering_cost, holding_cost_rate) ≈ 28.284271247461902
    end
end

@testset "Newsvendor" begin
    @test begin 
        horizon = 1

        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=1.0)
        customer = Customer("customer")
        
        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage)

        network = SupplyChain(1)
        
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)

        policy = OnHandUptoOrderingPolicy(0)

        initial_states = [State(; 
                                demand = Dict((customer, product) => rand(Poisson(10), horizon))) for i in 1:10]

        policies = Dict((l2, product) => policy)
        optimize!(network, policies, initial_states...)

        println(policy)

        final_states = [simulate(network, policies, initial_state) for initial_state in initial_states]

        println("lost sales: $(get_total_lost_sales(final_states[1]))")
        println("sales: $(get_total_sales(final_states[1]))")
        println("demand: $(get_total_demand(final_states[1]))")
        println("holding costs: $(get_total_holding_costs(final_states[1]))")
        true
    end
end

@testset "EOQ" begin
    @test begin 
        true
    end
end

@testset "Simulate" begin
    @test begin
        horizon = 50

        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=0.1)
        customer = Customer("customer")
        
        l1 = Lane(storage, customer)
        l2 = Lane(supplier, storage; fixed_cost=10, time=2)

        network = SupplyChain(horizon)
        
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)

        policy = NetSSOrderingPolicy(0, 0)

        initial_states = [State(; 
                                demand = Dict((customer, product) => rand(Poisson(10), horizon))) for i in 1:20]
                                
        policies = Dict((l2, product) => policy)
        for initial_state in initial_states
            simulate(network, policies, initial_state)
        end
        true
    end
end

include("docs.jl")
include("policy-tests.jl")
include("policy-cover-tests.jl")
include("policy-ss-tests.jl")
include("policy-beergame-tests.jl")