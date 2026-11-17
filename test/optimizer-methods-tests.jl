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

using Random

@testset "optimize! decompose=:product" begin
    # Several independent products, each stocked at one storage: the per-product search should end at
    # least as good as one joint search with the same total evaluation budget, and its groups should
    # be one per product.
    Random.seed!(11)
    horizon = 20
    nprod = 6
    products = [Product("p$i") for i in 1:nprod]
    supplier = Supplier("supplier")
    storage = Storage("storage")
    for p in products
        add_product!(storage, p; unit_holding_cost=0.2)
    end
    customer = Customer("customer")
    l1 = Lane(storage, customer)
    l2 = Lane(supplier, storage; time=2)
    means = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
    demands = [[rand(Poisson(means[i]), horizon) * 1.0 for i in 1:nprod] for _ in 1:5]
    n(s) = begin
        network = SupplyChain(horizon)
        add_supplier!(network, supplier); add_storage!(network, storage); add_customer!(network, customer)
        for p in products
            add_product!(network, p)
        end
        add_lane!(network, l1); add_lane!(network, l2)
        for (i, p) in enumerate(products)
            add_demand!(network, customer, p, demands[s][i]; sales_price=1.0, lost_sales_cost=1.0)
        end
        network
    end
    make_policies() = Dict((l2, p) => OnHandUptoOrderingPolicy(0) for p in products)
    cost(pol) = sum(metrics_cost_function(simulate(n(s), pol)) for s in 1:5)

    pol_joint = make_policies()
    optimize!(pol_joint, [n(s) for s in 1:5]...; method=:nelder_mead,
              nelder_mead_options=Dict{Symbol, Any}(:lower => 0.0, :upper => 1000.0, :maxfevals => 1200),
              cost_function=metrics_cost_function, record_history=false)
    pol_split = make_policies()
    optimize!(pol_split, [n(s) for s in 1:5]...; method=:nelder_mead, decompose=:product,
              nelder_mead_options=Dict{Symbol, Any}(:lower => 0.0, :upper => 1000.0, :maxfevals => 1200),
              cost_function=metrics_cost_function, record_history=false)
    @test cost(pol_split) <= cost(pol_joint) + 1e-6

    keys_sorted = sort(collect(keys(pol_split)); by = k -> (string(k[1]), k[2].name))
    pols = unique([pol_split[k] for k in keys_sorted])
    @test length(SupplyChainSimulation.product_groups(keys_sorted, pol_split, pols)) == nprod

    # one policy object shared by two products puts them in one group
    shared = OnHandUptoOrderingPolicy(0)
    pol_shared = Dict{Any, Any}((l2, p) => OnHandUptoOrderingPolicy(0) for p in products[3:end])
    pol_shared[(l2, products[1])] = shared
    pol_shared[(l2, products[2])] = shared
    ks = sort(collect(keys(pol_shared)); by = k -> (string(k[1]), k[2].name))
    @test length(SupplyChainSimulation.product_groups(ks, pol_shared, unique([pol_shared[k] for k in ks]))) == nprod - 1

    @test_throws ErrorException optimize!(make_policies(), n(1); decompose=:sku)
end
