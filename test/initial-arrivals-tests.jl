@testset "initial_arrivals: shipments already in transit arrive in their own period" begin
    # supplier -> storage (lead time 4) -> customer. A lane's initial_arrivals are
    # [time][destination] (see SupplyChainModeling.Lane), so a shipment due in
    # period 4 must be visible as in-transit stock from period 1 on - not only
    # shipments due in period 1.
    function build(arrivals; destinations=1)
        product = Product("product")
        supplier = Supplier("supplier")
        storages = [Storage("storage$i") for i in 1:destinations]
        customer = Customer("customer")
        for s in storages
            add_product!(s, product; initial_inventory=100)
        end
        network = SupplyChain(10)
        add_supplier!(network, supplier)
        foreach(s -> add_storage!(network, s), storages)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, Lane(storages[1], customer; unit_cost=0))
        inbound = destinations == 1 ?
            Lane(supplier, storages[1]; unit_cost=0, time=4, initial_arrivals=arrivals) :
            Lane(supplier, storages; unit_cost=0, times=fill(4, destinations), initial_arrivals=arrivals)
        add_lane!(network, inbound)
        add_demand!(network, customer, product, fill(0.0, 10); sales_price=1.0, lost_sales_cost=1.0)
        return network, storages, product
    end

    @testset "a shipment due after period 1 is not dropped" begin
        network, storages, product = build(Dict(Product("product") => [0, 0, 0, 40, 0, 0, 0, 0, 0, 0]))
        state = SupplyChainSimulation.State(network)
        # on hand 100 + 40 on the way
        @test get_net_inventory(state, storages[1], product, 1) == 140
        # ... and still counted at period 4, but no longer once it has landed (period 5)
        @test get_net_inventory(state, storages[1], product, 4) == 140
        @test get_net_inventory(state, storages[1], product, 5) == 100
    end

    @testset "shipments in several periods are all counted" begin
        arrivals = Dict(Product("product") => [5, 0, 7, 0, 0, 0, 0, 0, 0, 0])
        network, storages, product = build(arrivals)
        state = SupplyChainSimulation.State(network)
        @test get_net_inventory(state, storages[1], product, 1) == 112
        @test get_net_inventory(state, storages[1], product, 2) == 107   # the 5 has landed
    end

    @testset "several destinations: each gets its own shipments" begin
        # [time][destination]: period 2 brings 5 units to storage1 and 7 to storage2
        arrivals = Dict(Product("product") => [[0, 0], [5, 7], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        network, storages, product = build(arrivals; destinations=2)
        state = SupplyChainSimulation.State(network)
        @test get_net_inventory(state, storages[1], product, 1) == 105
        @test get_net_inventory(state, storages[2], product, 1) == 107
    end

    @testset "a shipment due in period 1 still works (the case the old indexing happened to handle)" begin
        network, storages, product = build(Dict(Product("product") => [10, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        state = SupplyChainSimulation.State(network)
        @test get_net_inventory(state, storages[1], product, 1) == 110
    end

    @testset "recommend_orders sees the in-transit stock" begin
        network, storages, product = build(Dict(Product("product") => [0, 0, 0, 40, 0, 0, 0, 0, 0, 0]))
        lane = only(l for l in network.lanes if l.origin isa Supplier)
        # order up to 200 net: 200 - (100 on hand + 40 in transit) = 60
        @test recommend_orders(network, Dict((lane, product) => NetUptoOrderingPolicy(200))) == Dict((lane, product) => 60)
    end
end
