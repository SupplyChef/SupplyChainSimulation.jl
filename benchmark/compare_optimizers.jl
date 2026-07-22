#=
Compares optimize!'s three search methods - :custom (the original hand-rolled
optimizer), :cma_es (CMAEvolutionStrategy.jl), and :nelder_mead (multi-start
Optim.jl NelderMead) - on the same simulation-optimization task, under a
matched evaluation budget, across several independent trials.

:nelder_mead was added after an earlier run of this script showed :cma_es
converging - and then plateauing - well before exhausting its evaluation
budget (see the convergence table this script prints): a deterministic local
search with restarts is a genuinely different approach from CMA-ES's
population-based one, and worth checking whether it matches CMA-ES's
solution quality in less time, or beats it by spending the leftover budget
on more restarts instead of one long population search.

What "better" means here is deliberately not just "lower final training
cost": a search that overfits a small fixed scenario sample can look great
on the scenarios it was tuned on and still be worse in practice. So this
mirrors test/policy-calibration-generalization-tests.jl's design - calibrate
on a training ensemble of demand regimes, then score the *held-out* ensemble
the policy was never tuned on - and reports both, plus wall-clock time (the
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

Budget/trial count come from COMPARE_MAXFEVALS/COMPARE_TRIALS env vars.

Also runnable via .github/workflows/compare-optimizers.yml.
=#

using Random
using Distributions: Normal, TDist, ccdf
using SupplyChainModeling
using SupplyChainSimulation

const CHECKPOINT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

function compare_optimizers_params()
    maxfevals = parse(Int, get(ENV, "COMPARE_MAXFEVALS", "3000"))
    trials = parse(Int, get(ENV, "COMPARE_TRIALS", "5"))
    return (; maxfevals, trials)
end

# Same scenario family as the calibration-generalization test: a single-stage
# newsvendor problem with a known closed-form optimum (the critical-ratio fill
# rate), so the training/held-out costs reported below have an external
# reference point, not just a relative "lower is better" comparison.
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

"""
    run_search(method, lane_policies, training_scenarios; maxfevals)

Reimplements optimize!'s body (see Optimization.jl) for `training_scenarios`
and `lane_policies`, but wraps the objective with an evaluation-count/
best-so-far tracker so the caller gets a convergence curve, not just the
final policy - then mutates `lane_policies`' policies in place exactly like
optimize! does. Returns `(elapsed_seconds, checkpoints)` where `checkpoints`
maps each fraction in CHECKPOINT_FRACTIONS to the best cost seen by that
point in the evaluation budget (forward-filled if the search stopped early).
"""
function run_search(method::Symbol, lane_policies, training_scenarios; maxfevals::Int, seed::Int)
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
        value = SupplyChainSimulation.minimize!(lane_policies, policies, collect(envs), collect(initial_states), x; cost_function=newsvendor_cost)
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
                 :SearchRange => (-0.0, 5000.0), :NumDimensions => length(x0)))
    elseif method === :cma_es
        best = SupplyChainSimulation.cma_es_optimize(tracked_f, x0,
            Dict{Symbol, Any}(:maxfevals => maxfevals, :seed => UInt(seed)))
    elseif method === :nelder_mead
        best = SupplyChainSimulation.nelder_mead_optimize(tracked_f, x0,
            Dict{Symbol, Any}(:maxfevals => maxfevals))
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

function main()
    (; maxfevals, trials) = compare_optimizers_params()
    methods = [:custom, :cma_es, :nelder_mead]

    println("Comparing optimize! methods: maxfevals=$maxfevals, trials=$trials")
    println("Training ensemble: $(length(TRAINING_REGIMES)) regimes x $TRAINING_REPLICATIONS replications = $(length(TRAINING_REGIMES) * TRAINING_REPLICATIONS) scenarios")
    println("Held-out ensemble: $(length(HELD_OUT_REGIMES)) regimes x $HELD_OUT_REPLICATIONS replications = $(length(HELD_OUT_REGIMES) * HELD_OUT_REPLICATIONS) scenarios")
    println("Target service level: $TARGET_SERVICE_LEVEL (critical-ratio newsvendor optimum)\n")

    results = Dict(m => (; wall_time=Float64[], training_cost=Float64[], held_out_cost=Float64[], fill_rate=Float64[], checkpoints=Vector{Float64}[]) for m in methods)

    for trial in 1:trials
        seed = 1000 + trial
        Random.seed!(seed)
        training_scenarios, supply_lane, product = build_ensemble(TRAINING_REGIMES, TRAINING_REPLICATIONS)

        for method in methods
            policy = BackwardCoverageOrderingPolicy([0.0, 0.0])
            lane_policies = Dict((supply_lane, product) => policy)

            elapsed, checkpoints = run_search(method, lane_policies, training_scenarios; maxfevals, seed)
            training_cost = checkpoints[end]

            Random.seed!(seed + 500_000) # held-out draws independent of training, but reproducible and identical across methods within a trial
            ho_cost, fill_rate = held_out_cost(policy, supply_lane, product, HELD_OUT_REGIMES, HELD_OUT_REPLICATIONS)

            push!(results[method].wall_time, elapsed)
            push!(results[method].training_cost, training_cost)
            push!(results[method].held_out_cost, ho_cost)
            push!(results[method].fill_rate, fill_rate)
            push!(results[method].checkpoints, checkpoints)

            println("trial=$trial method=$method wall_time=$(round(elapsed, digits=2))s training_cost=$(round(training_cost, digits=2)) held_out_cost=$(round(ho_cost, digits=2)) held_out_fill_rate=$(round(fill_rate, digits=4))")
        end
    end

    summary_lines = String[]
    push!(summary_lines, "## Optimizer comparison: " * join(methods, " vs "))
    push!(summary_lines, "")
    push!(summary_lines, "maxfevals=$maxfevals, trials=$trials")
    push!(summary_lines, "")
    push!(summary_lines, "### Per-method summary (mean ± std, median)")
    push!(summary_lines, "")
    push!(summary_lines, "| method | wall_time (s) | training_cost | held_out_cost | held_out_fill_rate |")
    push!(summary_lines, "|---|---|---|---|---|")
    for method in methods
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
    for method in methods
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
        for method in methods
            method === :custom && continue
            for (label, field) in (("training_cost", :training_cost), ("held_out_cost", :held_out_cost), ("wall_time (s)", :wall_time))
                diff, t, p = paired_t_test(getfield(results[:custom], field), getfield(results[method], field))
                push!(summary_lines, "| $method | $label | $(round(diff, digits=3)) | $(round(t, digits=3)) | $(round(p, digits=4)) |")
            end
        end
        push!(summary_lines, "")
        push!(summary_lines, "Negative diff means the alternative scored lower (better, since both costs are minimized); p < 0.05 suggests the difference isn't just trial-to-trial noise.")
    end

    summary = join(summary_lines, "\n")
    println("\n" * summary)

    summary_path = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
    if !isnothing(summary_path)
        open(summary_path, "a") do io
            println(io, summary)
        end
    end
end

main()
