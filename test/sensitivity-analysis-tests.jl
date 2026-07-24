using Random

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

@testset "sobol_indices matches the analytical Sobol indices of a linear function" begin
    @test begin
        # f(x) = x1 + 2*x2 over [0,1]^2 is purely additive (no interaction
        # term between x1 and x2), so its Sobol indices are known exactly:
        # S_i = c_i^2 * Var(x_i) / sum_j(c_j^2 * Var(x_j)). Both x1, x2 ~
        # Uniform(0,1) have equal variance, so this reduces to
        # S_i = c_i^2 / (c1^2 + c2^2): S1 = 1/5 = 0.2, S2 = 4/5 = 0.8 - and
        # since there is no interaction at all, ST == S1 exactly in the
        # infinite-sample limit. Seeded for a deterministic, non-flaky
        # comparison against a Monte Carlo estimator.
        Random.seed!(1)
        f = x -> x[1] + 2 * x[2]
        S1, ST = SupplyChainSimulation.sobol_indices(f, [0.0, 0.0], [1.0, 1.0]; samples=20000)

        isapprox(S1[1], 0.2; atol=0.03) && isapprox(S1[2], 0.8; atol=0.03) &&
            isapprox(ST[1], 0.2; atol=0.03) && isapprox(ST[2], 0.8; atol=0.03)
    end
end
