@testset "sensitivity_analysis" begin
    @test begin
        horizon = 1

        product = Product("product")

        supplier = Supplier("supplier")
        storage = Storage("storage")
        add_product!(storage, product; unit_holding_cost=1.0)
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

        original_parameters = [10.0, 20.0]
        policy = BackwardCoverageOrderingPolicy(copy(original_parameters))
        policies = Dict((l2, product) => policy)

        initial_states = [n() for i in 1:5]
        # Small :samples keeps this a fast smoke test - the default 1000 is
        # sized for a real analysis, not a unit test.
        result = sensitivity_analysis(policies, initial_states...;
                  options=Dict{Symbol, Any}(:lower => [0.0, 0.0], :upper => [100.0, 100.0], :samples => 4),
                  cost_function=metrics_cost_function, record_history=false)

        # A read-only diagnostic must not leave the policy perturbed to
        # whatever the last sampled point happened to be.
        get_parameters(policy) == original_parameters &&
            length(result.parameter_labels) == 2 &&
            length(result.S1) == 2 &&
            length(result.ST) == 2
    end
end
