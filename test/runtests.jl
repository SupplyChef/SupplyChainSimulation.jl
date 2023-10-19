using SupplyChainSimulation

using Distributions
using Test

@testset "Network" begin
    @test begin
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)

        product = Single("product")

        network = Network([], [storage], [customer], get_trips(l, 1), Single[product])

        get_sorted_locations(network) == [storage, customer]
    end

    @test begin
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(; origin = customer, destination = storage, unit_cost = 0)

        product = Single("product")

        network = Network([], [storage], [customer], get_trips(l, 1), [product])

        get_sorted_locations(network) == [customer, storage]
    end

    @test begin
        customer = Customer("c")
        storage = Storage("s")
        storage2 = Storage("s2")
        
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)
        l2 = Lane(; origin = storage2, destination = storage, unit_cost = 0)

        product = Single("product")

        network = Network([], [storage, storage2], [customer], get_trips([l, l2], 1), [product])

        get_sorted_locations(network) == [storage2, storage, customer]
    end

    @test begin
        customer = Customer("c")
        storage = Storage("s")
        storage2 = Storage("s2")
        
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)
        l2 = Lane(; origin = storage2, destination = storage, unit_cost = 0)

        product = Single("product")

        network = Network([], [storage, storage2], [customer], get_trips([l, l2], 1), [product])

        get_downstream_customers(network, storage) == [customer] && get_downstream_customers(network, storage2) == [customer]
    end
end

@testset "State" begin
    @test begin
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)

        product = Single("product")

        network = Network([], [storage], [customer], get_trips([l], 1), [product])

        initial_state = State(; on_hand_inventory = Dict(storage => Dict(product => 0)), 
                                in_transit_inventory = Dict(storage => Dict(product => [10, 0]), customer => Dict(product => [0, 0])),
                                demand = Dict((customer, product) => [0, 0]),
                                policies = Dict((l, product) => OnHandUptoOrderingPolicy(0)))

        get_inbound_orders(initial_state, storage, product, 1) == 0 && 
        get_outbound_orders(initial_state, storage, product, 1) == 0 &&
        get_net_inventory(initial_state, storage, product, 1) == 10
    end

    @test begin
        customer = Customer("c")
        storage = Storage("s")
        storage2 = Storage("s2")
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)
        l2 = Lane(; origin = storage2, destination = storage, unit_cost = 0)

        product = Single("product")

        o = Order(0, l, [(product, 5)], 1)
        o2 = Order(0, l2, [(product, 15)], 1)

        network = Network([], [storage, storage2], [customer], get_trips([l, l2], 1), [product])

        initial_state = State(; on_hand_inventory = Dict(storage => Dict(product => 0)), 
                                in_transit_inventory = Dict(storage => Dict(product => [10, 0]), storage2 => Dict(product => [100, 0]), customer => Dict(product => [0, 0])), 
                                pending_outbound_order_lines = Dict(storage => o.lines, storage2 => o2.lines),
                                demand = Dict((customer, product) => [0, 0]),
                                policies = Dict((l, product) => OnHandUptoOrderingPolicy(0)))

        #println("inbound $(get_inbound_orders(initial_state, storage, product, 1))") 
        #println("outbound $(get_outbound_orders(initial_state, storage, product, 1))")
        #println("net $(get_net_inventory(initial_state, storage, product, 1))")
                            
        get_inbound_orders(initial_state, storage, product, 1) == 15 && 
        get_outbound_orders(initial_state, storage, product, 1) == 5 &&
        get_net_inventory(initial_state, storage, product, 1) == 20
    end

    @test begin
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)

        p = Single("product")

        network = Network([], [storage], [customer], get_trips([l], 2), [p])

        initial_state = State(; on_hand_inventory = Dict(storage => Dict(p => 0)), 
                                in_transit_inventory = Dict(storage => Dict(p => [10, 0]), customer => Dict(p => [0, 0])),
                                demand = Dict((customer, p) => [0, 0]),
                                policies = Dict((l, p) => OnHandUptoOrderingPolicy(0)))

        final_state = simulate(network, 2, initial_state)

        final_state.on_hand_inventory[storage][p] == 10 && 
        length(collect(Base.Iterators.flatten(final_state.historical_orders))) == 0 && 
        length(collect(Base.Iterators.flatten(final_state.historical_filled_order_lines))) == 0
    end

    @test begin
        customer = Customer("c")
        storage = Storage("s")
        l = Lane(; origin = storage, destination = customer, unit_cost = 0)

        p = Single("product")

        network = Network([], [storage], [customer], get_trips([l], 2), [p])

        initial_state = State(; on_hand_inventory = Dict(storage => Dict(p => 10)), 
                                in_transit_inventory = Dict(storage => Dict(p => [10, 0]), customer => Dict(p => [0, 0])),
                                demand = Dict((customer, p) => [10, 10]),
                                policies = Dict((l, p) => OnHandUptoOrderingPolicy(0)))

        final_state = simulate(network, 2, initial_state)

        println(get_total_holding_costs(final_state))

        final_state.on_hand_inventory[storage][p] == 0 && 
        length(collect(Base.Iterators.flatten(final_state.historical_orders))) == 2 && 
        length(collect(Base.Iterators.flatten(final_state.historical_filled_order_lines))) == 2
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

        product = Single("product")

        supplier = Supplier("supplier")
        storage = Storage("storage", Dict(product => 1.0))
        customer = Customer("customer")
        
        l1 = Lane(; origin = storage, destination = customer)
        l2 = Lane(; origin = supplier, destination = storage)

        network = Network([supplier], [storage], [customer], get_trips([l1, l2], horizon), [product])

        policy = OnHandUptoOrderingPolicy(0)

        initial_states = [State(; on_hand_inventory = Dict(storage => Dict(product => 0)), 
                                demand = Dict((customer, product) => rand(Poisson(10), horizon)),
                                policies = Dict((l2, product) => policy)) for i in 1:10]

        optimize!(network, horizon, initial_states...)

        println(policy)

        final_states = [simulate(network, horizon, initial_state) for initial_state in initial_states]

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

        product = Single("product")

        supplier = Supplier("supplier")
        storage = Storage("storage", Dict(product => 0.1))
        customer = Customer("customer")
        
        l1 = Lane(; origin=storage, destination=customer)
        l2 = Lane(; origin=supplier, destination=storage, fixed_cost=10, lead_time=2)

        network = Network([supplier], [storage], [customer], get_trips([l1, l2], horizon), [product])

        policy = NetSSOrderingPolicy(0, 0)

        initial_states = [State(; on_hand_inventory = Dict(storage => Dict(product => 0)), 
                                demand = Dict((customer, product) => rand(Poisson(10), horizon)),
                                policies = Dict((l2, product) => policy)) for i in 1:20]

        for initial_state in initial_states
            simulate(network, horizon, initial_state)
        end
        true
    end
end

include("policy-tests.jl")
include("policy-cover-tests.jl")
include("policy-ss-tests.jl")
include("policy-beergame-tests.jl")