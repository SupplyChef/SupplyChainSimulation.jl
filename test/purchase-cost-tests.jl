@testset "Purchase costs" begin
    function build(horizon, unit_cost)
        product = Product("product")
        supplier = Supplier("supplier")
        add_product!(supplier, product; unit_cost=unit_cost)
        storage = Storage("storage")
        customer = Customer("customer")
        l1 = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=0)
        network = SupplyChain(horizon)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)
        add_demand!(network, customer, product, fill(0.0, horizon); sales_price=1.0, lost_sales_cost=1.0)
        return network, l2, product
    end

    @testset "accrued at placement: quantity * supplier unit_cost" begin
        network, l2, product = build(1, 10.0)
        final_state = simulate(network, Dict((l2, product) => QuantityOrderingPolicy([100])))
        @test final_state.metrics.purchase_costs ≈ 1000.0
        @test get_total_purchase_costs(final_state) ≈ 1000.0
    end

    @testset "origin with no unit_cost for the product accrues nothing" begin
        network, l2, product = build(1, 0.0)
        final_state = simulate(network, Dict((l2, product) => QuantityOrderingPolicy([100])))
        @test final_state.metrics.purchase_costs == 0.0
    end

    @testset "profit_cost_function adds purchases; metrics_cost_function unchanged" begin
        network, l2, product = build(1, 10.0)
        final_state = simulate(network, Dict((l2, product) => QuantityOrderingPolicy([100])))
        @test profit_cost_function(final_state) ≈ metrics_cost_function(final_state) + 1000.0
    end
end
