@testset "ResponseProfile validation" begin
    @test begin
        p = ResponseProfile(0.25, 0.25, 0.25, 0.25)
        p.wait_share == 0.25 && p.substitute_share == 0.25 &&
            p.buy_elsewhere_share == 0.25 && p.defect_share == 0.25
    end

    @test begin
        p = ResponseProfile(1.0, 0.0, 0.0, 0.0)
        p.wait_share == 1.0
    end

    @test_throws ArgumentError ResponseProfile(0.5, 0.5, 0.5, 0.5)
    @test_throws ArgumentError ResponseProfile(-0.1, 0.5, 0.5, 0.1)
    @test_throws ArgumentError ResponseProfile(0.3, 0.3, 0.3, 0.3)
end

# All four scenarios below share the same shape of network - a Customer
# fed by a single Storage - built fresh per @test via n() so no state leaks
# across tests. Every ResponseProfile below is a degenerate (one share = 1)
# distribution: the point isn't to test rand() (that's Base's job), it's to
# deterministically pin down exactly which mechanism each outcome triggers,
# so every assertion here is a hand-traceable exact value, not a
# statistical one.
@testset "ResponseProfile: wait behaves like a per-pair backlog" begin
    # storage starts with 0 stock; a single 10-unit resupply lands at
    # period 3. demand is a single 10-unit pulse at period 1, then nothing -
    # so the *only* way period 1's order is ever filled is if it's still
    # alive (backlogged) when that period-3 resupply lands.
    n = () -> begin
        product = Product("product")
        customer = Customer("customer")
        storage = Storage("storage")
        add_product!(storage, product)
        storage2 = Storage("storage2")
        add_product!(storage2, product)

        l = Lane(storage, customer; unit_cost=0)
        l2 = Lane(storage2, storage; unit_cost=0, initial_arrivals=Dict(product => [0, 0, 10, 0]))

        network = SupplyChain(4)
        add_storage!(network, storage)
        add_storage!(network, storage2)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)
        add_demand!(network, customer, product, [10.0, 0.0, 0.0, 0.0]; sales_price=1.0, lost_sales_cost=1.0)

        return (network, customer, product, storage)
    end
    policies = Dict{Tuple{Lane, Product}, InventoryOrderingPolicy}()

    @test begin
        # Baseline: default customer_backlog=false, no profile - the order
        # is dropped as lost the period after it's created and the
        # period-3 resupply then just sits unconsumed.
        (network, customer, product, storage) = n()
        final_state = simulate(network, policies)
        get_total_demand(final_state) == 10.0 &&
            get_total_sales(final_state) == 0.0 &&
            get_total_lost_sales(final_state) == 10.0 &&
            get_on_hand_inventory(final_state, storage, product) == 10.0
    end

    @test begin
        # customer_backlog=true (network-wide): the order queues from the
        # moment it's created and is filled once the resupply lands.
        (network, customer, product, storage) = n()
        final_state = simulate(network, policies; customer_backlog=true)
        get_total_demand(final_state) == 10.0 &&
            get_total_sales(final_state) == 10.0 &&
            get_total_lost_sales(final_state) == 0.0 &&
            final_state.metrics.backlog == 20.0
    end

    @test begin
        # Per-pair wait_share=1.0: same fulfillment outcome as
        # customer_backlog=true above (the order is still filled by the
        # period-3 resupply) via the exact same due_date=typemax(Int64)
        # mechanism - just decided per-pair, one period later (only once
        # the order is first confirmed unfilled, not at creation), hence
        # the smaller backlog-cost total.
        (network, customer, product, storage) = n()
        profiles = Dict((customer, product) => ResponseProfile(1.0, 0.0, 0.0, 0.0))
        final_state = simulate(network, policies; response_profiles=profiles)
        get_total_demand(final_state) == 10.0 &&
            get_total_sales(final_state) == 10.0 &&
            get_total_lost_sales(final_state) == 0.0 &&
            final_state.metrics.backlog == 10.0
    end

    @test begin
        # Per-pair substitute_share=1.0: mechanically identical to the
        # no-profile default above - lost this period, resupply unconsumed.
        (network, customer, product, storage) = n()
        profiles = Dict((customer, product) => ResponseProfile(0.0, 1.0, 0.0, 0.0))
        final_state = simulate(network, policies; response_profiles=profiles)
        get_total_demand(final_state) == 10.0 &&
            get_total_sales(final_state) == 0.0 &&
            get_total_lost_sales(final_state) == 10.0 &&
            get_on_hand_inventory(final_state, storage, product) == 10.0
    end

    @test begin
        # Per-pair buy_elsewhere_share=1.0: same as substitute above - this
        # package has no product-substitution data model, so the two
        # outcomes are indistinguishable mechanically.
        (network, customer, product, storage) = n()
        profiles = Dict((customer, product) => ResponseProfile(0.0, 0.0, 1.0, 0.0))
        final_state = simulate(network, policies; response_profiles=profiles)
        get_total_demand(final_state) == 10.0 &&
            get_total_sales(final_state) == 0.0 &&
            get_total_lost_sales(final_state) == 10.0
    end
end

@testset "ResponseProfile: defect zeroes future demand" begin
    # No resupply at all - storage stays at 0 stock for the whole run - and
    # demand pulses every period, so every order is a stockout. Under
    # defect_share=1.0 the pair should stop generating orders entirely
    # after its first stockout is confirmed; under substitute_share=1.0 (or
    # the default) it keeps generating - and losing - a fresh order every
    # period.
    n = () -> begin
        product = Product("product")
        customer = Customer("customer")
        storage = Storage("storage")
        add_product!(storage, product)
        l = Lane(storage, customer; unit_cost=0)

        network = SupplyChain(4)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_demand!(network, customer, product, [10.0, 10.0, 10.0, 10.0]; sales_price=1.0, lost_sales_cost=1.0)

        return (network, customer, product)
    end
    policies = Dict{Tuple{Lane, Product}, InventoryOrderingPolicy}()

    @test begin
        (network, customer, product) = n()
        profiles = Dict((customer, product) => ResponseProfile(0.0, 1.0, 0.0, 0.0))
        final_state = simulate(network, policies; response_profiles=profiles)
        # Demand never stops: 4 periods x 10 units, all lost.
        get_total_demand(final_state) == 40.0 &&
            get_total_sales(final_state) == 0.0 &&
            get_total_lost_sales(final_state) == 40.0
    end

    @test begin
        (network, customer, product) = n()
        profiles = Dict((customer, product) => ResponseProfile(0.0, 0.0, 0.0, 1.0))
        final_state = simulate(network, policies; response_profiles=profiles)
        # Period 1's order stockouts; that's confirmed during period 2's
        # processing, by which point period 2's own order was already
        # placed (place_orders runs before send_inventory! each period) -
        # so exactly 2 periods' worth of demand (20 units) are ever
        # generated, and periods 3-4 place no order at all.
        get_total_demand(final_state) == 20.0 &&
            get_total_sales(final_state) == 0.0 &&
            get_total_lost_sales(final_state) == 20.0
    end
end

@testset "ResponseProfile: reusing a State/Env across simulate() calls (as optimize! does)" begin
    # optimize! builds one Env and calls simulate(env, policies, state)
    # repeatedly, reset!(state)-ing between trials rather than rebuilding
    # State/Env from scratch. A defection recorded in one trial must not
    # leak into the next, or every trial after the first stockout would
    # see permanently-suppressed demand for no reason tied to that trial.
    @test begin
        product = Product("product")
        customer = Customer("customer")
        storage = Storage("storage")
        add_product!(storage, product)
        l = Lane(storage, customer; unit_cost=0)

        network = SupplyChain(4)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_demand!(network, customer, product, [10.0, 10.0, 10.0, 10.0]; sales_price=1.0, lost_sales_cost=1.0)

        policies = Dict{Tuple{Lane, Product}, InventoryOrderingPolicy}()
        profiles = Dict((customer, product) => ResponseProfile(0.0, 0.0, 0.0, 1.0))

        state = State(network)
        env = Env(network, [state], policies; response_profiles=profiles)

        simulate(env, policies, state)
        first_run_demand = get_total_demand(state)
        first_run_defected = state.defected[state.location_index[customer], state.product_index[product]]

        reset!(state)
        reset_clears_defection = !state.defected[state.location_index[customer], state.product_index[product]]

        simulate(env, policies, state)
        second_run_demand = get_total_demand(state)

        first_run_demand == 20.0 && first_run_defected && reset_clears_defection && second_run_demand == 20.0
    end
end
