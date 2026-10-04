@testset "Cash: payment terms, capital costs, peak cash outlay" begin
    # supplier -> storage (lead time 1) -> customer, horizon 4. Orders of 100 units at 10 each in
    # period 1; the customer buys 40 units at 25 in period 3.
    function build(; terms=PaymentTerms(), cost_of_capital=0.0, horizon=4, lane_cost=0.0, cash_budget=Inf)
        product = Product("product")
        supplier = Supplier("supplier"; payment_terms=terms)
        add_product!(supplier, product; unit_cost=10.0)
        storage = Storage("storage")
        customer = Customer("customer")
        l1 = Lane(storage, customer; unit_cost=0)
        l2 = Lane(supplier, storage; unit_cost=lane_cost, time=1)
        network = SupplyChain(horizon; cost_of_capital=cost_of_capital, cash_budget=cash_budget)
        add_supplier!(network, supplier)
        add_storage!(network, storage)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l1)
        add_lane!(network, l2)
        demand = zeros(horizon)
        horizon >= 3 && (demand[3] = 40.0)
        add_demand!(network, customer, product, demand; sales_price=25.0, lost_sales_cost=0.0)
        return network, l2, product
    end
    run(network, l2, product) = simulate(network, Dict((l2, product) => QuantityOrderingPolicy([100, 0, 0, 0])))

    @testset "default terms pay in full at order; no cost of capital" begin
        state = run(build()...)
        @test get_cash_out(state) ≈ [1000.0, 0.0, 0.0, 0.0]
        @test get_cash_in(state) ≈ [0.0, 0.0, 1000.0, 0.0]
        @test sum(get_cash_out(state)) ≈ state.metrics.purchase_costs
        @test sum(get_cash_in(state)) ≈ state.metrics.sales
        @test get_peak_cash_outlay(state) ≈ 1000.0
        @test get_total_capital_costs(state) == 0.0
        # cash does not change the existing cost functions
        @test profit_cost_function(state) ≈ metrics_cost_function(state) + 1000.0
    end

    @testset "30% deposit at order, balance at shipment" begin
        state = run(build(terms=PaymentTerms(deposit_share=0.3))...)
        # the order ships the period it is placed: 300 at order + 700 at shipment, both in period 1
        @test get_cash_out(state) ≈ [1000.0, 0.0, 0.0, 0.0]
    end

    @testset "balance one period after shipment (credit terms)" begin
        state = run(build(terms=PaymentTerms(deposit_share=0.3, balance_offset=1))...)
        @test get_cash_out(state) ≈ [300.0, 700.0, 0.0, 0.0]
        @test get_peak_cash_outlay(state) ≈ 1000.0   # 1000 spent by period 2, customer pays in period 3
    end

    @testset "balance before shipment is not paid earlier than the order" begin
        state = run(build(terms=PaymentTerms(deposit_share=0.3, balance_offset=-2))...)
        @test get_cash_out(state) ≈ [1000.0, 0.0, 0.0, 0.0]
    end

    @testset "a balance due after the horizon is not in the curve" begin
        state = run(build(terms=PaymentTerms(deposit_share=0.3, balance_offset=10))...)
        @test get_cash_out(state) ≈ [300.0, 0.0, 0.0, 0.0]
        @test get_peak_cash_outlay(state) ≈ 300.0
    end

    @testset "freight and tariffs are cash out at shipment" begin
        state = run(build(lane_cost=2.0)...)
        # 100 units at 2 on the supplier lane in period 1
        @test get_cash_out(state) ≈ [1200.0, 0.0, 0.0, 0.0]
    end

    @testset "capital costs: rate times cumulative net cash out, per period, floored at 0" begin
        state = run(build(cost_of_capital=0.01)...)
        # cumulative net cash out at the end of each period: 1000, 1000, 0 (sales 1000), 0
        @test get_total_capital_costs(state) ≈ 0.01 * (1000.0 + 1000.0)
        # later payment means less capital tied up
        later = run(build(terms=PaymentTerms(deposit_share=0.3, balance_offset=1), cost_of_capital=0.01)...)
        @test get_total_capital_costs(later) ≈ 0.01 * (300.0 + 1000.0)
        @test get_total_capital_costs(later) < get_total_capital_costs(state)
    end

    @testset "cash_cost_function: budget penalty" begin
        state = run(build(cost_of_capital=0.01)...)
        base = profit_cost_function(state) + get_total_capital_costs(state)
        @test cash_cost_function()(state) ≈ base
        @test cash_cost_function(budget=1000.0)(state) ≈ base
        @test cash_cost_function(budget=800.0)(state) ≈ base + 200.0
        @test cash_cost_function(budget=800.0, penalty=3.0)(state) ≈ base + 600.0
        # the budget defaults to the supply chain's own
        budgeted = run(build(cost_of_capital=0.01, cash_budget=800.0)...)
        @test cash_cost_function()(budgeted) ≈ base + 200.0
        @test cash_cost_function(budget=Inf)(budgeted) ≈ base
    end

    @testset "reset! clears cash, and re-simulation reproduces it" begin
        network, l2, product = build(terms=PaymentTerms(deposit_share=0.3, balance_offset=1), cost_of_capital=0.01)
        policies = Dict((l2, product) => QuantityOrderingPolicy([100, 0, 0, 0]))
        initial_state = State(network)
        env = Env(network, [initial_state], policies)
        state = simulate(env, policies, initial_state)
        first_out = get_cash_out(state); first_peak = get_peak_cash_outlay(state)
        reset!(state)
        @test all(iszero, state.metrics.cash_out) && state.metrics.capital_costs == 0.0 && state.metrics.peak_cash_outlay == 0.0
        state = simulate(env, policies, state)
        @test get_cash_out(state) ≈ first_out && get_peak_cash_outlay(state) ≈ first_peak
    end

    @testset "metrics agree with and without history" begin
        network, l2, product = build(terms=PaymentTerms(deposit_share=0.3, balance_offset=1), cost_of_capital=0.01)
        policies = Dict((l2, product) => QuantityOrderingPolicy([100, 0, 0, 0]))
        a = State(network); b = State(network)
        sa = simulate(Env(network, [a], policies; record_history=true), policies, a)
        sb = simulate(Env(network, [b], policies; record_history=false), policies, b)
        @test get_cash_out(sa) == get_cash_out(sb) && get_peak_cash_outlay(sa) == get_peak_cash_outlay(sb)
    end
end
