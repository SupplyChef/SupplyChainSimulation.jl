#=
Compares optimize!'s four search methods - :custom (the original hand-rolled
optimizer), :cma_es (CMAEvolutionStrategy.jl), :nelder_mead (multi-start
Optim.jl NelderMead), and :bayesopt (BayesianOptimization.jl) - on two
structurally different simulation-optimization tasks, under a matched
evaluation budget, across several independent trials.

:nelder_mead was added after an earlier run of this script showed :cma_es
converging - and then plateauing - well before exhausting its evaluation
budget (see the convergence table this script prints): a deterministic local
search with restarts is a genuinely different approach from CMA-ES's
population-based one, and worth checking whether it matches CMA-ES's
solution quality in less time, or beats it by spending the leftover budget
on more restarts instead of one long population search.

:bayesopt was added for the same underlying reason - :cma_es plateauing well
short of its budget suggests these problems' actual information content is
exhausted far earlier than thousands of evaluations - but pursues a
different fix: instead of spending the same number of expensive real
evaluations more cleverly, use a cheap Gaussian-process surrogate to need
far fewer of them in the first place. Worth trying specifically because
every evaluation here is a full discrete-event simulation (expensive) over a
small number of parameters (2-6 so far) - exactly the regime where trading
surrogate-fitting cost for real evaluations pays off.

Two problems are compared, not just one:
- "newsvendor": test/policy-calibration-generalization-tests.jl's single-stage
  problem, 2 parameters (BackwardCoverageOrderingPolicy on one lane), a known
  closed-form optimum (the critical-ratio fill rate).
- "beer_game": test/policy-beergame-tests.jl's 4-echelon problem, 6 parameters
  (BackwardCoverageOrderingPolicy on 3 lanes, each with lead time), no
  closed-form optimum. Included because newsvendor's 2-parameter search space
  is small enough that all three methods already land on statistically
  indistinguishable solutions (see the first run's results) - a weak signal
  for picking an algorithm. A higher-dimensional, multi-echelon problem is a
  more honest test of whether algorithm choice actually matters on quality,
  not just speed.

What "better" means here is deliberately not just "lower final training
cost": a search that overfits a small fixed scenario sample can look great
on the scenarios it was tuned on and still be worse in practice. So both
problems calibrate on a training ensemble, then score a *held-out* ensemble
the policy was never tuned on - and report both, plus wall-clock time (the
algorithms don't necessarily cost the same per evaluation) and a paired
comparison across trials against the :custom baseline (same trial => same
scenarios => same starting point for every method, so differences are
attributable to the algorithm, not to scenario variance).

This intentionally does not call optimize! itself: comparing convergence
*during* the search (not just the final answer) needs a per-evaluation hook
that optimize!'s public API doesn't expose, so this reimplements optimize!'s
setup (Env construction, sorted policy keys, x0 assembly - see Optimization.jl)
with an instrumented objective, then calls the same internal
SupplyChainSimulation.bboptimize/cma_es_optimize/nelder_mead_optimize entry
points optimize! uses.

Run with:
    julia --project=benchmark benchmark/compare_optimizers.jl

Budget/trial count come from COMPARE_MAXFEVALS/COMPARE_TRIALS env vars,
shared across both problems.

Also runnable via .github/workflows/compare-optimizers.yml.
=#

using Random
using Distributions: Poisson, Normal, TDist, ccdf
using SupplyChainModeling
using SupplyChainSimulation

const CHECKPOINT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
const METHODS = [:custom, :cma_es, :nelder_mead, :bayesopt]

function compare_optimizers_params()
    maxfevals = parse(Int, get(ENV, "COMPARE_MAXFEVALS", "3000"))
    trials = parse(Int, get(ENV, "COMPARE_TRIALS", "5"))
    # Default (5000.0) matches every *_optimize function's own built-in
    # default, so leaving this unset reproduces prior behavior exactly.
    # Overridable to test whether that shared default - inherited from
    # :custom's original DE SearchRange - is simply too large relative to
    # where these policies' real parameters live (see nelder_mead_optimize's
    # docstring: an earlier version of *that* method had the identical
    # failure mode from the same oversized box, fixed by anchoring restarts
    # near x0 instead of sampling the full box uniformly - :bayesopt's Sobol
    # design and :cma_es's sigma0 default aren't anchored that way, and both
    # regressed badly on beer_game's 6-parameter box in the first real
    # comparison run).
    upper = parse(Float64, get(ENV, "COMPARE_UPPER", "5000.0"))
    return (; maxfevals, trials, upper)
end

"""
    run_search(method, lane_policies, training_scenarios, cost_function; maxfevals, seed, upper)

Reimplements optimize!'s body (see Optimization.jl) for `training_scenarios`
and `lane_policies`, but wraps `cost_function` with an evaluation-count/
best-so-far tracker so the caller gets a convergence curve, not just the
final policy - then mutates `lane_policies`' policies in place exactly like
optimize! does. Returns `(elapsed_seconds, checkpoints)` where `checkpoints`
maps each fraction in CHECKPOINT_FRACTIONS to the best cost seen by that
point in the evaluation budget (forward-filled if the search stopped early).

Generic over `lane_policies`' shape (one (lane, product) pair or several) and
over `cost_function` - neither problem-specific detail affects the search
mechanics themselves.

`upper` (paired with a fixed lower bound of 0.0) is passed to every method
explicitly rather than relying on each *_optimize function's own default, so
compare_optimizers_params's COMPARE_UPPER override actually reaches all four
methods identically - a controlled, single-variable test of whether the
shared box size itself explains :bayesopt/:cma_es's regressions.
"""
function run_search(method::Symbol, lane_policies, training_scenarios, cost_function::Function; maxfevals::Int, seed::Int, upper::Float64=5000.0)
    initial_states = SupplyChainSimulation.State.(training_scenarios)
    envs = [SupplyChainSimulation.Env(sc, initial_states, lane_policies; record_history=true) for sc in training_scenarios]

    sorted_keys = sort(collect(keys(lane_policies)); by = k -> (string(k[1]), k[2].name))
    policies = unique([lane_policies[k] for k in sorted_keys])
    x0 = convert(Array{Float64, 1}, vcat([SupplyChainSimulation.get_parameters(policy) for policy in policies]...))

    eval_count = Ref(0)
    best_so_far = Ref(Inf)
    checkpoint_evals = [max(1, round(Int, frac * maxfevals)) for frac in CHECKPOINT_FRACTIONS]
    checkpoints = fill(NaN, length(CHECKPOINT_FRACTIONS))

    tracked_f = function (x)
        value = SupplyChainSimulation.minimize!(lane_policies, policies, collect(envs), collect(initial_states), x; cost_function=cost_function)
        eval_count[] += 1
        if value < best_so_far[]
            best_so_far[] = value
        end
        for (i, ce) in enumerate(checkpoint_evals)
            if eval_count[] == ce && isnan(checkpoints[i])
                checkpoints[i] = best_so_far[]
            end
        end
        return value
    end

    Random.seed!(seed)
    start = time()
    if method === :custom
        best = SupplyChainSimulation.bboptimize(tracked_f, x0,
            Dict(:MaxFuncEvals => maxfevals, :MaxStepsWithoutProgress => maxfevals,
                 :SearchRange => (-0.0, upper), :NumDimensions => length(x0)))
    elseif method === :cma_es
        best = SupplyChainSimulation.cma_es_optimize(tracked_f, x0,
            Dict{Symbol, Any}(:maxfevals => maxfevals, :seed => UInt(seed), :upper => upper))
    elseif method === :nelder_mead
        best = SupplyChainSimulation.nelder_mead_optimize(tracked_f, x0,
            Dict{Symbol, Any}(:maxfevals => maxfevals, :upper => upper))
    elseif method === :bayesopt
        # Same maxfevals as every other method, for the same reason they all
        # get it here (a fair, matched-budget comparison) - even though
        # bayesopt_optimize's own default (200) is deliberately much smaller.
        # Whether Bayesian optimization is still worthwhile once forced up to
        # the other methods' scale (its own per-step Gaussian-process fit
        # cost grows with the number of points observed) is itself part of
        # what this comparison is meant to show, not something to hide by
        # quietly giving it an easier budget than the others.
        best = SupplyChainSimulation.bayesopt_optimize(tracked_f, x0,
            Dict{Symbol, Any}(:maxfevals => maxfevals, :upper => upper))
    else
        error("run_search: unknown method $method")
    end
    elapsed = time() - start

    # Forward-fill any checkpoint the search never reached (early stop) with
    # the best value known by the end, so trials that stop early still
    # contribute a full row to the convergence table instead of a NaN gap.
    for i in eachindex(checkpoints)
        if isnan(checkpoints[i])
            checkpoints[i] = best_so_far[]
        end
    end

    i = 1
    for policy in policies
        SupplyChainSimulation.set_parameters!(policy, best[i:i+length(SupplyChainSimulation.get_parameters(policy))-1])
        i = i + length(SupplyChainSimulation.get_parameters(policy))
    end

    return elapsed, checkpoints
end

"""
    held_out_cost(cost_function, build_held_out_replication, count)

Generic held-out evaluator: calls the zero-argument `build_held_out_replication`
closure `count` times, each call expected to return a fresh
`(scenario, held_out_policies, product)` - a newly-built scenario (its own
independent random draw), a policies dict pairing this scenario's own lanes
with the already-trained policy objects, and the product to track fill rate
for. Building a fresh matching dict per replication (rather than reusing the
training dict directly against a structurally-equal-but-different scenario)
mirrors the newsvendor-specific held_out_cost method below, so both problems
get held-out evaluation the same, obviously-correct way.
"""
function held_out_cost(cost_function::Function, build_held_out_replication::Function, count::Int)
    total_ordered = 0.0
    total_filled = 0.0
    total_cost = 0.0
    for _ in 1:count
        sc, held_out_policies, product = build_held_out_replication()
        state = simulate(sc, held_out_policies)

        orders = filter(ol -> ol.destination isa Customer && ol.product == product, collect(Base.Iterators.flatten(state.historical_orders)))
        filled = filter(ol -> ol.destination isa Customer && ol.product == product, collect(Base.Iterators.flatten(state.historical_filled_orders)))
        total_ordered += sum(ol -> ol.quantity, orders; init=0)
        total_filled += sum(ol -> ol.quantity, filled; init=0)
        total_cost += cost_function(state)
    end
    fill_rate = total_ordered > 0 ? total_filled / total_ordered : NaN
    return total_cost, fill_rate
end

function mean_std(xs)
    m = sum(xs) / length(xs)
    v = length(xs) > 1 ? sum((x - m)^2 for x in xs) / (length(xs) - 1) : 0.0
    return m, sqrt(v)
end

function median_of(xs)
    s = sort(xs)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# Paired (two-sided) t-test on trial-by-trial differences: valid here because
# both methods see the *same* scenarios in a given trial (only the RNG stream
# consumed by the search itself differs), so per-trial scenario variance
# cancels out of the difference instead of adding noise to an unpaired test.
function paired_t_test(a, b)
    diffs = a .- b
    n = length(diffs)
    m, s = mean_std(diffs)
    if s == 0.0
        return m, NaN, m == 0.0 ? 1.0 : 0.0
    end
    t = m / (s / sqrt(n))
    p = 2 * ccdf(TDist(n - 1), abs(t))
    return m, t, p
end

function summarize_results(problem_name::String, maxfevals::Int, trials::Int, results)
    summary_lines = String[]
    push!(summary_lines, "## $problem_name: " * join(METHODS, " vs "))
    push!(summary_lines, "")
    push!(summary_lines, "maxfevals=$maxfevals, trials=$trials")
    push!(summary_lines, "")
    push!(summary_lines, "### Per-method summary (mean ± std, median)")
    push!(summary_lines, "")
    push!(summary_lines, "| method | wall_time (s) | training_cost | held_out_cost | held_out_fill_rate |")
    push!(summary_lines, "|---|---|---|---|---|")
    for method in METHODS
        r = results[method]
        wt_m, wt_s = mean_std(r.wall_time)
        tc_m, tc_s = mean_std(r.training_cost)
        hc_m, hc_s = mean_std(r.held_out_cost)
        fr_m, fr_s = mean_std(r.fill_rate)
        push!(summary_lines, "| $method | $(round(wt_m, digits=2)) ± $(round(wt_s, digits=2)) (median $(round(median_of(r.wall_time), digits=2))) " *
                              "| $(round(tc_m, digits=2)) ± $(round(tc_s, digits=2)) (median $(round(median_of(r.training_cost), digits=2))) " *
                              "| $(round(hc_m, digits=2)) ± $(round(hc_s, digits=2)) (median $(round(median_of(r.held_out_cost), digits=2))) " *
                              "| $(round(fr_m, digits=4)) ± $(round(fr_s, digits=4)) |")
    end

    push!(summary_lines, "")
    push!(summary_lines, "### Convergence (mean best-so-far training cost at each fraction of the eval budget)")
    push!(summary_lines, "")
    push!(summary_lines, "| method | " * join(["$(Int(100*f))%" for f in CHECKPOINT_FRACTIONS], " | ") * " |")
    push!(summary_lines, "|---|" * "---|"^length(CHECKPOINT_FRACTIONS))
    for method in METHODS
        r = results[method]
        means = [mean_std([c[i] for c in r.checkpoints])[1] for i in eachindex(CHECKPOINT_FRACTIONS)]
        push!(summary_lines, "| $method | " * join([string(round(v, digits=2)) for v in means], " | ") * " |")
    end

    if trials > 1
        push!(summary_lines, "")
        push!(summary_lines, "### Paired comparison against :custom across trials (two-sided paired t-test)")
        push!(summary_lines, "")
        push!(summary_lines, "| alternative | metric | mean diff (custom - alternative) | t | p |")
        push!(summary_lines, "|---|---|---|---|---|")
        for method in METHODS
            method === :custom && continue
            for (label, field) in (("training_cost", :training_cost), ("held_out_cost", :held_out_cost), ("wall_time (s)", :wall_time))
                diff, t, p = paired_t_test(getfield(results[:custom], field), getfield(results[method], field))
                push!(summary_lines, "| $method | $label | $(round(diff, digits=3)) | $(round(t, digits=3)) | $(round(p, digits=4)) |")
            end
        end
        push!(summary_lines, "")
        push!(summary_lines, "Negative diff means the alternative scored lower (better, since both costs are minimized); p < 0.05 suggests the difference isn't just trial-to-trial noise.")
    end

    return summary_lines
end

# =============================================================================
# Problem 1: newsvendor (test/policy-calibration-generalization-tests.jl)
# =============================================================================

# Single-stage newsvendor problem with a known closed-form optimum (the
# critical-ratio fill rate), so the training/held-out costs reported below
# have an external reference point, not just a relative "lower is better"
# comparison.
const HORIZON = 100
const LEAD_TIME = 3
const TARGET_SERVICE_LEVEL = 0.95
const HOLDING_COST = 0.1
const LOST_SALES_COST = HOLDING_COST * TARGET_SERVICE_LEVEL / (1 - TARGET_SERVICE_LEVEL)

const TRAINING_REGIMES = [(10.0, 2.0), (20.0, 5.0), (30.0, 7.0), (15.0, 6.0)]
const TRAINING_REPLICATIONS = 4
const HELD_OUT_REGIMES = [(25.0, 6.0), (12.0, 4.0)]
const HELD_OUT_REPLICATIONS = 5

function build_scenario(mean_demand, demand_std)
    product = Product("p1")
    customer = Customer("c1")
    storage = Storage("s1"; initial_opened=true)
    add_product!(storage, product; unit_holding_cost=HOLDING_COST)
    supplier = Supplier("sup1")
    add_product!(supplier, product; unit_cost=1.0)

    sc = SupplyChain(HORIZON)
    add_product!(sc, product)
    add_customer!(sc, customer)
    add_storage!(sc, storage)
    add_supplier!(sc, supplier)

    demand = [max(0.0, round(rand(Normal(mean_demand, demand_std)))) for _ in 1:HORIZON]
    add_demand!(sc, customer, product, demand; lost_sales_cost=LOST_SALES_COST, sales_price=0.0)

    add_lane!(sc, Lane(storage, customer; unit_cost=0.0))
    supply_lane = Lane(supplier, storage; unit_cost=0.0, time=LEAD_TIME)
    add_lane!(sc, supply_lane)

    return sc, supply_lane, product
end

newsvendor_cost(state) = get_total_lost_sales(state) + get_total_holding_costs(state)

function build_ensemble(regimes, replications)
    scenarios = []
    supply_lane = nothing
    product = nothing
    for (mean_demand, demand_std) in regimes, _ in 1:replications
        sc, lane, p = build_scenario(mean_demand, demand_std)
        push!(scenarios, sc)
        supply_lane = lane
        product = p
    end
    return scenarios, supply_lane, product
end

# Existing (pre-beer_game) held-out evaluator: kept exactly as-is (including
# its own copy of the total_ordered/filled/cost loop) rather than routed
# through the generic method above, to avoid any risk of regressing the
# newsvendor comparison that's already been validated in CI. The generic
# `held_out_cost(cost_function, build_held_out_replication, count)` method
# above is a distinct-enough signature that both coexist via multiple
# dispatch with no ambiguity.
function held_out_cost(policy, supply_lane, product, regimes, replications)
    total_ordered = 0.0
    total_filled = 0.0
    total_cost = 0.0
    for (mean_demand, demand_std) in regimes, _ in 1:replications
        sc, lane, p = build_scenario(mean_demand, demand_std)
        held_out_policies = Dict((lane, p) => policy)
        state = simulate(sc, held_out_policies)

        orders = filter(ol -> ol.destination isa Customer && ol.product == p, collect(Base.Iterators.flatten(state.historical_orders)))
        filled = filter(ol -> ol.destination isa Customer && ol.product == p, collect(Base.Iterators.flatten(state.historical_filled_orders)))
        total_ordered += sum(ol -> ol.quantity, orders; init=0)
        total_filled += sum(ol -> ol.quantity, filled; init=0)
        total_cost += newsvendor_cost(state)
    end
    fill_rate = total_ordered > 0 ? total_filled / total_ordered : NaN
    return total_cost, fill_rate
end

function run_newsvendor_comparison(maxfevals::Int, trials::Int, upper::Float64)
    println("Training ensemble: $(length(TRAINING_REGIMES)) regimes x $TRAINING_REPLICATIONS replications = $(length(TRAINING_REGIMES) * TRAINING_REPLICATIONS) scenarios")
    println("Held-out ensemble: $(length(HELD_OUT_REGIMES)) regimes x $HELD_OUT_REPLICATIONS replications = $(length(HELD_OUT_REGIMES) * HELD_OUT_REPLICATIONS) scenarios")
    println("Target service level: $TARGET_SERVICE_LEVEL (critical-ratio newsvendor optimum)\n")

    results = Dict(m => (; wall_time=Float64[], training_cost=Float64[], held_out_cost=Float64[], fill_rate=Float64[], checkpoints=Vector{Float64}[]) for m in METHODS)

    for trial in 1:trials
        seed = 1000 + trial
        Random.seed!(seed)
        training_scenarios, supply_lane, product = build_ensemble(TRAINING_REGIMES, TRAINING_REPLICATIONS)

        for method in METHODS
            policy = BackwardCoverageOrderingPolicy([0.0, 0.0])
            lane_policies = Dict((supply_lane, product) => policy)

            elapsed, checkpoints = run_search(method, lane_policies, training_scenarios, newsvendor_cost; maxfevals, seed, upper)
            training_cost = checkpoints[end]

            Random.seed!(seed + 500_000) # held-out draws independent of training, but reproducible and identical across methods within a trial
            ho_cost, fill_rate = held_out_cost(policy, supply_lane, product, HELD_OUT_REGIMES, HELD_OUT_REPLICATIONS)

            push!(results[method].wall_time, elapsed)
            push!(results[method].training_cost, training_cost)
            push!(results[method].held_out_cost, ho_cost)
            push!(results[method].fill_rate, fill_rate)
            push!(results[method].checkpoints, checkpoints)

            println("newsvendor trial=$trial method=$method wall_time=$(round(elapsed, digits=2))s training_cost=$(round(training_cost, digits=2)) held_out_cost=$(round(ho_cost, digits=2)) held_out_fill_rate=$(round(fill_rate, digits=4))")
        end
    end

    return summarize_results("newsvendor", maxfevals, trials, results)
end

# =============================================================================
# Problem 2: beer_game (test/policy-beergame-tests.jl)
# =============================================================================

# 4-echelon problem (supplier -> factory -> wholesaler -> retailer ->
# customer), 3 BackwardCoverageOrderingPolicy policies (6 parameters total,
# one per lead-time-bearing lane) - a higher-dimensional, structurally
# different problem from newsvendor's 2-parameter single-stage one. Mirrors
# test/policy-beergame-tests.jl's beer_game() exactly, parametrized so
# training/held-out ensembles can be built independently.
const BEERGAME_HORIZON = 200
const BEERGAME_HOLDING_COST = 0.1
const BEERGAME_TRAINING_COUNT = 20
const BEERGAME_HELD_OUT_COUNT = 10

function build_beergame_scenario()
    product = Product("beer")
    customer = Customer("customer")
    retailer = Storage("retailer")
    add_product!(retailer, product; unit_holding_cost=BEERGAME_HOLDING_COST, initial_inventory=20)
    wholesaler = Storage("wholesaler")
    add_product!(wholesaler, product; unit_holding_cost=BEERGAME_HOLDING_COST, initial_inventory=20)
    factory = Storage("factory")
    add_product!(factory, product; unit_holding_cost=BEERGAME_HOLDING_COST, initial_inventory=20)
    supplier = Supplier("supplier")

    sc = SupplyChain(BEERGAME_HORIZON)
    add_supplier!(sc, supplier)
    add_storage!(sc, retailer)
    add_storage!(sc, wholesaler)
    add_storage!(sc, factory)
    add_customer!(sc, customer)
    add_product!(sc, product)

    l1 = Lane(retailer, customer; unit_cost=0.0)
    l2 = Lane(wholesaler, retailer; unit_cost=0.0, time=2)
    l3 = Lane(factory, wholesaler; unit_cost=0.0, time=2)
    l4 = Lane(supplier, factory; unit_cost=0.0, time=4)
    add_lane!(sc, l1)
    add_lane!(sc, l2)
    add_lane!(sc, l3)
    add_lane!(sc, l4)

    demand = rand(Poisson(10), BEERGAME_HORIZON) * 1.0
    add_demand!(sc, customer, product, demand; sales_price=1.0, lost_sales_cost=1.0)

    return sc, l2, l3, l4, product
end

function build_beergame_ensemble(count)
    scenarios = SupplyChain[]
    l2 = l3 = l4 = nothing
    product = nothing
    for _ in 1:count
        sc, l2_, l3_, l4_, p = build_beergame_scenario()
        push!(scenarios, sc)
        l2, l3, l4, product = l2_, l3_, l4_, p
    end
    return scenarios, l2, l3, l4, product
end

function run_beergame_comparison(maxfevals::Int, trials::Int, upper::Float64)
    println("\nTraining ensemble: $BEERGAME_TRAINING_COUNT scenarios (Poisson(10) demand)")
    println("Held-out ensemble: $BEERGAME_HELD_OUT_COUNT scenarios (fresh Poisson(10) draws)")
    println("Cost function: metrics_cost_function (sales_price=1.0 here, unlike newsvendor's 0.0, so revenue is part of the objective)\n")

    results = Dict(m => (; wall_time=Float64[], training_cost=Float64[], held_out_cost=Float64[], fill_rate=Float64[], checkpoints=Vector{Float64}[]) for m in METHODS)

    for trial in 1:trials
        seed = 2000 + trial
        Random.seed!(seed)
        training_scenarios, l2, l3, l4, product = build_beergame_ensemble(BEERGAME_TRAINING_COUNT)

        for method in METHODS
            policy2 = BackwardCoverageOrderingPolicy([0.0, 0.0])
            policy3 = BackwardCoverageOrderingPolicy([0.0, 0.0])
            policy4 = BackwardCoverageOrderingPolicy([0.0, 0.0])
            lane_policies = Dict((l2, product) => policy2, (l3, product) => policy3, (l4, product) => policy4)

            elapsed, checkpoints = run_search(method, lane_policies, training_scenarios, metrics_cost_function; maxfevals, seed, upper)
            training_cost = checkpoints[end]

            Random.seed!(seed + 500_000)
            build_replication = function ()
                sc, l2_ho, l3_ho, l4_ho, p_ho = build_beergame_scenario()
                held_out_policies = Dict((l2_ho, p_ho) => policy2, (l3_ho, p_ho) => policy3, (l4_ho, p_ho) => policy4)
                return sc, held_out_policies, p_ho
            end
            ho_cost, fill_rate = held_out_cost(metrics_cost_function, build_replication, BEERGAME_HELD_OUT_COUNT)

            push!(results[method].wall_time, elapsed)
            push!(results[method].training_cost, training_cost)
            push!(results[method].held_out_cost, ho_cost)
            push!(results[method].fill_rate, fill_rate)
            push!(results[method].checkpoints, checkpoints)

            println("beer_game trial=$trial method=$method wall_time=$(round(elapsed, digits=2))s training_cost=$(round(training_cost, digits=2)) held_out_cost=$(round(ho_cost, digits=2)) held_out_fill_rate=$(round(fill_rate, digits=4))")
        end
    end

    return summarize_results("beer_game", maxfevals, trials, results)
end

function main()
    (; maxfevals, trials, upper) = compare_optimizers_params()

    println("Comparing optimize! methods: maxfevals=$maxfevals, trials=$trials, upper=$upper\n")

    println("=== newsvendor ===")
    newsvendor_summary = run_newsvendor_comparison(maxfevals, trials, upper)

    println("\n=== beer_game ===")
    beergame_summary = run_beergame_comparison(maxfevals, trials, upper)

    summary = join(vcat(newsvendor_summary, [""], beergame_summary), "\n")
    println("\n" * summary)

    summary_path = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
    if !isnothing(summary_path)
        open(summary_path, "a") do io
            println(io, summary)
        end
    end
end

main()
