@testset "optimize! method=:cma_es" begin
    # Smoke-tests optimize!'s :cma_es path end to end (CMAEvolutionStrategy.jl
    # wiring in Optimization.jl's cma_es_optimize) - the same shape as the
    # :custom Newsvendor test in runtests.jl, since no exact numeric outcome
    # depends on either search's trajectory, only that it runs cleanly and
    # leaves a usable policy behind.
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

        policy = OnHandUptoOrderingPolicy(0)
        policies = Dict((l2, product) => policy)

        initial_states = [n() for i in 1:5]
        # Small maxfevals keeps this a fast smoke test - the default 15000
        # is sized for benchmark/compare_optimizers.jl, not a unit test.
        # This exercises cma_es_optimize's default (scalar) bounds and its
        # seed option-forwarding branch.
        optimize!(policies, initial_states...;
                  method=:cma_es,
                  cma_es_options=Dict{Symbol, Any}(:maxfevals => 200, :seed => UInt(42)),
                  cost_function=metrics_cost_function, record_history=false)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]
        true
    end

    @test begin
        # Exercises cma_es_optimize's vector-bounds and popsize option-
        # forwarding branches, which the scalar-bounds test above doesn't
        # reach.
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

        policy = BackwardCoverageOrderingPolicy([0.0, 0.0])
        policies = Dict((l2, product) => policy)

        initial_states = [n() for i in 1:5]
        optimize!(policies, initial_states...;
                  method=:cma_es,
                  cma_es_options=Dict{Symbol, Any}(:lower => [0.0, 0.0], :upper => [100.0, 100.0], :maxfevals => 200, :popsize => 8),
                  cost_function=metrics_cost_function, record_history=false)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]
        true
    end

    @test_throws ErrorException begin
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
        add_demand!(network, customer, product, [10.0]; sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict((l2, product) => OnHandUptoOrderingPolicy(0))
        optimize!(policies, network; method=:not_a_real_method)
    end
end

@testset "optimize! method=:nelder_mead" begin
    # Smoke-tests optimize!'s :nelder_mead path end to end (Optim.jl wiring in
    # Optimization.jl's nelder_mead_optimize) - same shape/reasoning as the
    # :cma_es tests above.
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

        policy = OnHandUptoOrderingPolicy(0)
        policies = Dict((l2, product) => policy)

        initial_states = [n() for i in 1:5]
        # Small maxfevals/restarts keeps this a fast smoke test - exercises
        # nelder_mead_optimize's default (scalar) bounds path.
        optimize!(policies, initial_states...;
                  method=:nelder_mead,
                  nelder_mead_options=Dict{Symbol, Any}(:maxfevals => 200, :restarts => 2),
                  cost_function=metrics_cost_function, record_history=false)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]
        true
    end

    @test begin
        # Exercises nelder_mead_optimize's vector-bounds path, which the
        # scalar-bounds test above doesn't reach.
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

        policy = BackwardCoverageOrderingPolicy([0.0, 0.0])
        policies = Dict((l2, product) => policy)

        initial_states = [n() for i in 1:5]
        optimize!(policies, initial_states...;
                  method=:nelder_mead,
                  nelder_mead_options=Dict{Symbol, Any}(:lower => [0.0, 0.0], :upper => [100.0, 100.0], :maxfevals => 200, :restarts => 2),
                  cost_function=metrics_cost_function, record_history=false)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]
        true
    end
end

@testset "optimize! method=:bayesopt" begin
    # Smoke-tests optimize!'s :bayesopt path end to end (BayesianOptimization.jl/
    # GaussianProcesses.jl wiring in Optimization.jl's bayesopt_optimize) - same
    # shape/reasoning as the :cma_es/:nelder_mead tests above.
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

        policy = OnHandUptoOrderingPolicy(0)
        policies = Dict((l2, product) => policy)

        initial_states = [n() for i in 1:5]
        # Small maxfevals keeps this a fast smoke test - the default 200 is
        # already small (bayesopt_optimize isn't meant to run thousands of
        # evaluations), but this exercises its default (scalar) bounds path
        # with an even smaller budget.
        optimize!(policies, initial_states...;
                  method=:bayesopt,
                  bayesopt_options=Dict{Symbol, Any}(:maxfevals => 30),
                  cost_function=metrics_cost_function, record_history=false)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]
        true
    end

    @test begin
        # Exercises bayesopt_optimize's vector-bounds (n > 1, so points are
        # Tuples internally, not plain Float64) and num_new_samples option-
        # forwarding branches, which the scalar-bounds test above doesn't
        # reach.
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

        policy = BackwardCoverageOrderingPolicy([0.0, 0.0])
        policies = Dict((l2, product) => policy)

        initial_states = [n() for i in 1:5]
        optimize!(policies, initial_states...;
                  method=:bayesopt,
                  bayesopt_options=Dict{Symbol, Any}(:lower => [0.0, 0.0], :upper => [100.0, 100.0], :maxfevals => 30, :num_new_samples => 20),
                  cost_function=metrics_cost_function, record_history=false)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]
        true
    end
end

@testset "optimize! method=:custom with custom_options surrogate seeding" begin
    # Smoke-tests optimize!'s custom_options=>:surrogate_seed_samples path
    # (quadratic_surrogate_optimum wiring in Optimization.jl) - same
    # shape/reasoning as the other method tests above. Also exercises the
    # disabled-by-default path implicitly: every other test in this file
    # calls :custom (via runtests.jl's Newsvendor test, or other testsets)
    # without custom_options at all and still passes, confirming
    # surrogate_seed_samples defaulting to 0 changes nothing for them.
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

        policy = BackwardCoverageOrderingPolicy([0.0, 0.0])
        policies = Dict((l2, product) => policy)

        initial_states = [n() for i in 1:5]
        # 2 parameters -> quadratic_surrogate_optimum needs at least
        # 1 + 2*2 + 2*1/2 = 6 samples; 10 leaves a small safety margin while
        # keeping this a fast smoke test. :SearchRange isn't overridden here
        # (params is Dict{Symbol,Float64} - a Tuple value wouldn't fit) so
        # this exercises the default (-0.0, 5000.0) box.
        optimize!(policies, initial_states...;
                  method=:custom,
                  params=Dict(:MaxFuncEvals => 50.0, :MaxStepsWithoutProgress => 50.0),
                  custom_options=Dict{Symbol, Any}(:surrogate_seed_samples => 10),
                  cost_function=metrics_cost_function, record_history=false)

        final_states = [simulate(initial_state, policies) for initial_state in initial_states]
        true
    end
end
