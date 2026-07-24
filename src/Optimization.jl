
#import Distributions
using Dates

"""
    metrics_cost_function(s::State)

An alternative to hand-rolling `s -> -get_total_sales(s) + get_total_lost_sales(s) + ...`
as a `cost_function` for `optimize!`: reads `s.metrics` (see `SimMetrics`) instead of
scanning `s`'s `historical_*` arrays the way the equivalent `get_total_*` functions in
Reporting.jl do. Since `SimMetrics` is always kept up to date regardless of
`Env.record_history`, this is the cost function to pass (together with
`record_history=false`) to get `optimize!`'s full per-trial-simulation speedup.

Not `optimize!`'s default: its default is left exactly as it always was (scanning
history, with `record_history` defaulting to `true`) so existing callers - including
ones whose `cost_function` reads `get_total_*` directly - see byte-for-byte identical
behavior. Summing the same costs in a different order (event order here, Set/Dict
iteration order for the `get_total_*` scan) is only equal up to floating-point
rounding, not bit-for-bit, and `optimize!`'s search is sensitive enough to that
- many float comparisons deciding which candidate "wins" over thousands of
evaluations - that swapping the default out from under existing callers could shift
what they converge to. Opt in explicitly once you don't need exact reproducibility
against a pre-existing baseline.
"""
metrics_cost_function(s) = -s.metrics.sales + s.metrics.lost_sales + s.metrics.holding_costs + s.metrics.trip_fixed_costs + s.metrics.trip_unit_costs + 0.001 * s.metrics.orders

function minimize!(lane_policies, policies, envs::Array{Env, 1}, initial_states::Array{State, 1}, x::AbstractVector{Float64}; cost_function)
    i = 1
    for policy in policies
        set_parameters!(policy, x[i:i+length(get_parameters(policy))-1])
        i = i + length(get_parameters(policy))
    end

    value = 0
    for i in 1:length(initial_states)
        # Reset the state's mutable containers in place rather than paying for a
        # fresh deepcopy of the (read-only) supply chain on every evaluation -
        # simulate() then simulates initial_states[i] directly and returns it.
        reset!(initial_states[i])
        final_state = simulate(envs[i], lane_policies, initial_states[i])

        #println(final_state)

        #println("$x $(get_total_lost_sales(final_state)) $(get_total_orders(final_state))")
        value += cost_function(final_state)
    end
    return value
end

"""
    optimize!(supplychain::SupplyChain, lane_policies, initial_states...; cost_function, record_history, method, cma_es_options)

    Optimizes the inventory policies in the supply chain by simulating the inventory movement starting from the initial states and costing the results with the cost function.

    `record_history` controls whether each of the (typically thousands of) trial
    simulations `optimize!` runs archives its full per-period history (see
    `Env.record_history`). It defaults to `true`, matching every trial simulation's
    behavior before `record_history` existed, so existing callers - including custom
    `cost_function`s that call the history-scanning `get_total_*` functions in
    Reporting.jl - keep working unchanged. Pass `record_history=false` once your
    `cost_function` only reads `state.metrics` (see `SimMetrics` and
    `metrics_cost_function`) to skip that per-trial bookkeeping.

    This is independent of `BackwardCoverageOrderingPolicy`: it reads
    `state.outbound_order_quantities` instead of the history-scanning
    `get_total_*` functions, and that index is maintained regardless of
    `record_history` (see `required_lookback`/`Env.needs_outbound_order_index`),
    so `record_history=false` is safe with that policy too.

    `method` selects the search algorithm used to minimize `cost_function` over the
    policies' parameters:
    - `:custom` (the default): the original hand-rolled differential-evolution-style
      optimizer in this file (`bboptimize`/`params`). Kept as the default so existing
      callers see byte-for-byte identical behavior.
    - `:cma_es`: Covariance Matrix Adaptation Evolution Strategy, via the
      CMAEvolutionStrategy.jl package. Tuned via `cma_es_options` (see below) instead
      of `params`, since CMA-ES's options aren't all `Float64` (bounds are vectors,
      some are integers) and `params` is typed `Dict{Symbol, Float64}`.
    - `:nelder_mead`: multi-start Nelder-Mead simplex search, via Optim.jl. Tuned via
      `nelder_mead_options` (see below). A deliberately different kind of algorithm
      from the other two (a deterministic local search with restarts, vs. their
      population-based stochastic search) - worth trying because
      benchmark/compare_optimizers.jl showed :cma_es converging (and then plateauing)
      well before exhausting its evaluation budget on the newsvendor benchmark,
      suggesting a cheaper local refinement with restarts to escape local optima
      might match its quality in even less time, or find a better optimum with the
      leftover budget spent on more restarts instead of one long population search.
    - `:bayesopt`: Bayesian optimization (Kriging surrogate + Expected Improvement
      acquisition function), via Surrogates.jl. Tuned via `bayesopt_options` (see
      below). Worth trying specifically because every
      evaluation here is a full discrete-event simulation (expensive) over a small
      number of parameters (2-6 in the benchmarks so far) - exactly the regime
      where trading cheap surrogate-model fitting for far fewer real evaluations
      pays off, unlike the other three methods, which all spend thousands of real
      evaluations even though :cma_es's own convergence curve suggests the actual
      information content of these problems is exhausted far earlier.

    `cma_es_options` (only used when `method === :cma_es`) accepts:
    - `:lower`, `:upper`: box constraints, each either a scalar (broadcast to every
      parameter) or a per-parameter vector. Default `0.0`/`5000.0`, matching
      `:custom`'s default `:SearchRange`.
    - `:maxfevals`: total evaluation budget across all restarts combined. Default
      `15000`, matching `:custom`'s default `:MaxFuncEvals`, so the methods are
      comparable under the same budget.
    - `:sigma0`: initial global step size (CMA-ES's `s0`). Defaults to a quarter of
      the bounds' range, a commonly-used rule of thumb when there's no prior on
      where in the box the optimum sits.
    - `:restarts`: number of *additional* CMA-ES runs beyond the first, each with a
      bigger population than the last (IPOP-CMA-ES - see `cma_es_optimize`'s
      docstring for why). Default `3`. Each gets an even share of `:maxfevals`.
    - `:incpopsize`: population-size multiplier applied per restart. Default `2`.
    - `:popsize`, `:seed`, `:verbosity`: passed straight through to
      `CMAEvolutionStrategy.minimize` when given; otherwise left at that package's
      own defaults (`:verbosity` defaults to `0` here instead, since `:custom`
      already prints its own progress and CMA-ES's is redundant in that context).
      `:popsize`, if given, is the *first* restart's population size - later
      restarts still scale up from it by `:incpopsize` each time.

    `bayesopt_options` (only used when `method === :bayesopt`) accepts:
    - `:lower`, `:upper`: box constraints, same shape/defaults as `cma_es_options`.
    - `:maxfevals`: total real simulation-evaluation budget, split between the
      initial space-filling design and the guided acquisition-function search (see
      `:initializer_iterations`). Default `200` - deliberately nowhere near the
      other methods' `15000`: Bayesian optimization trades a much smaller number of
      expensive real evaluations for the cost of fitting/querying a Kriging
      surrogate at every step, so giving it thousands of evaluations would make
      that per-step surrogate cost (which grows with the number of points
      observed) dominate, not help - see `bayesopt_optimize`'s docstring.
    - `:initializer_iterations`: how many of `:maxfevals` are spent on an initial
      space-filling (Sobol sequence) design before the acquisition-guided search
      begins, rather than on `:maxfevals` itself. Default `min(5 * length(x0),
      maxfevals ÷ 2)`.
    - `:num_new_samples`: how many candidate points Expected Improvement
      evaluates *on the cheap surrogate* per real iteration before picking the
      single best one to actually cost with `f` - this does not consume any of
      `:maxfevals` itself (see `bayesopt_optimize`'s docstring for why only one
      real evaluation happens per iteration regardless of this value). Default
      `100`.

    `nelder_mead_options` (only used when `method === :nelder_mead`) accepts:
    - `:lower`, `:upper`: box constraints, same shape/defaults as `cma_es_options`.
      Plain Nelder-Mead is unconstrained, so out-of-box candidates are clamped back
      into range before being costed, rather than using Optim.jl's `Fminbox` (a
      log-barrier wrapper built for gradient-based methods, not a natural fit for a
      derivative-free simplex search).
    - `:maxfevals`: total evaluation budget across all restarts combined. Default
      `15000`, matching the other two methods.
    - `:restarts`: number of *additional* Nelder-Mead runs beyond the one seeded at
      `x0`, each getting an even share of `:maxfevals`. Default `5` (6 runs total).
      Each restart perturbs the current incumbent best by a random step (see
      `:restart_scale`), not a fresh uniformly random point in the whole box - see
      `nelder_mead_optimize`'s docstring for why a global-random-restart version of
      this was actively harmful on a 6-parameter problem.
    - `:restart_scale`: restart perturbation size, as a fraction of `:upper - :lower`
      in each dimension. Default `0.2`.
"""
function optimize!(lane_policies, supplychains...; params::Dict{Symbol, Float64}=Dict{Symbol, Float64}(), cost_function=s->-get_total_sales(s) + get_total_lost_sales(s) + get_total_holding_costs(s) + get_total_trip_fixed_costs(s) + get_total_trip_unit_costs(s) + 0.001 * get_total_orders(s), record_history::Bool=true, method::Symbol=:custom, cma_es_options::Dict{Symbol, Any}=Dict{Symbol, Any}(), nelder_mead_options::Dict{Symbol, Any}=Dict{Symbol, Any}(), bayesopt_options::Dict{Symbol, Any}=Dict{Symbol, Any}())
    initial_states = State.(supplychains)
    envs = [Env(supplychain, initial_states, lane_policies; record_history=record_history) for supplychain in supplychains]

    # Iterate lane_policies in a deterministic order (by lane/product name)
    # rather than raw Dict key order: keys(lane_policies) iterates in hash
    # bucket order, which for id-hashed Lane/Product keys depends on
    # construction order across the whole process, not just this network's
    # structure. Since this order determines which slice of the optimizer's
    # parameter vector gets assigned to which policy (see minimize! above),
    # letting it vary non-reproducibly changes which local optimum the
    # search converges to.
    sorted_keys = sort(collect(keys(lane_policies)); by = k -> (string(k[1]), k[2].name))
    policies = unique([lane_policies[k] for k in sorted_keys])
    #println(policies)

    x0 = vcat([get_parameters(policy) for policy in policies]...)
    x0 = convert(Array{Float64, 1}, x0)

    f = x -> minimize!(lane_policies, policies, collect(envs), collect(initial_states), x; cost_function=cost_function)

    if method === :custom
        best = SupplyChainSimulation.bboptimize(f,
                         x0,
                        merge(Dict(:MaxFuncEvals => 15000,
                             :MaxStepsWithoutProgress => 1500,
                             :SearchRange => (-0.0, 5000.0),
                             :NumDimensions => length(x0)), params))
    elseif method === :cma_es
        best = cma_es_optimize(f, x0, cma_es_options)
    elseif method === :nelder_mead
        best = nelder_mead_optimize(f, x0, nelder_mead_options)
    elseif method === :bayesopt
        best = bayesopt_optimize(f, x0, bayesopt_options)
    else
        error("optimize!: unknown method $(repr(method)); supported methods are :custom, :cma_es, :nelder_mead, and :bayesopt")
    end

    i = 1
    for policy in policies
        set_parameters!(policy, best[i:i+length(get_parameters(policy))-1])
        i = i + length(get_parameters(policy))
    end
end

"""
    sensitivity_analysis(lane_policies, supplychains...; cost_function, record_history, options)

Runs a global (Sobol) sensitivity analysis of `cost_function` against every
policy parameter across `lane_policies`, via GlobalSensitivity.jl - a
different question from `optimize!`'s "what's the best policy?": "which
policy parameters actually matter for cost?" Useful as a triage step before
spending an `optimize!` budget tuning parameters that barely move the
outcome, or for understanding *why* a policy landed where it did.

Shares `optimize!`'s setup (Env construction, sorted policy keys, x0
assembly, the same `minimize!`-based evaluator - see `optimize!`'s comment
on why `sorted_keys`' order has to be deterministic) rather than reimplementing
it, so the parameter ordering `S1`/`ST` are reported in exactly matches what
`optimize!` itself searches over.

Restores every policy's original parameters before returning:  `minimize!`
mutates each policy's parameters as a side effect of evaluating a candidate
`x` (see its definition above) - without explicitly restoring `x0` afterward,
this function, despite being a read-only diagnostic, would otherwise leave
every policy set to whatever the last of GlobalSensitivity.jl's many sampled
points happened to be, rather than left unchanged as a caller would expect.

Returns a NamedTuple `(parameter_labels, S1, ST)`: `parameter_labels[i]`
identifies which policy (by the lane(s)/product it's attached to) and which
parameter index within that policy the `i`'th entry of `S1`/`ST` (first-order
and total-order Sobol indices) belongs to. A parameter's `S1` close to 0
means it barely affects `cost_function` on its own within the given bounds;
`ST` also captures its interactions with other parameters, so `ST >> S1` for
a parameter flags that it mostly matters *together with* another one, not on
its own.

`options` accepts:
- `:lower`, `:upper`: box constraints, each either a scalar (broadcast to
  every parameter) or a per-parameter vector - same shape/defaults
  (`0.0`/`5000.0`) as `optimize!`'s other `*_options` dicts.
- `:samples`: controls evaluation cost - GlobalSensitivity.jl's Sobol method
  needs `samples * (2 * length(x0) + 2)` real evaluations (some of the
  `2 * length(x0) + 2` factor is inherent to how Sobol indices are
  estimated, not a tunable). Default `1000`.
"""
function sensitivity_analysis(lane_policies, supplychains...; cost_function=s->-get_total_sales(s) + get_total_lost_sales(s) + get_total_holding_costs(s) + get_total_trip_fixed_costs(s) + get_total_trip_unit_costs(s) + 0.001 * get_total_orders(s), record_history::Bool=true, options::Dict{Symbol, Any}=Dict{Symbol, Any}())
    initial_states = State.(supplychains)
    envs = [Env(supplychain, initial_states, lane_policies; record_history=record_history) for supplychain in supplychains]

    sorted_keys = sort(collect(keys(lane_policies)); by = k -> (string(k[1]), k[2].name))
    policies = unique([lane_policies[k] for k in sorted_keys])

    x0 = convert(Array{Float64, 1}, vcat([get_parameters(policy) for policy in policies]...))
    n = length(x0)

    parameter_labels = String[]
    for policy in policies
        matching_keys = ["$(lane) / $(product.name)" for (lane, product) in sorted_keys if lane_policies[(lane, product)] === policy]
        label_prefix = join(matching_keys, ", ")
        for k in 1:length(get_parameters(policy))
            push!(parameter_labels, "$label_prefix param $k")
        end
    end

    lower = get(options, :lower, 0.0)
    upper = get(options, :upper, 5000.0)
    lower_bounds = lower isa AbstractVector ? convert(Array{Float64, 1}, lower) : fill(convert(Float64, lower), n)
    upper_bounds = upper isa AbstractVector ? convert(Array{Float64, 1}, upper) : fill(convert(Float64, upper), n)
    samples = get(options, :samples, 1000)

    f = x -> minimize!(lane_policies, policies, collect(envs), collect(initial_states), x; cost_function=cost_function)

    p_range = [[lower_bounds[i], upper_bounds[i]] for i in 1:n]
    result = GlobalSensitivity.gsa(f, GlobalSensitivity.Sobol(), p_range; samples=samples)

    i = 1
    for policy in policies
        set_parameters!(policy, x0[i:i+length(get_parameters(policy))-1])
        i = i + length(get_parameters(policy))
    end

    return (parameter_labels = parameter_labels, S1 = result.S1, ST = result.ST)
end

"""
    cma_es_optimize(f, x0, options)

Minimizes `f` starting from `x0` with CMAEvolutionStrategy.jl, translating this
package's `optimize!(...; method=:cma_es, cma_es_options=...)` options (see
`optimize!`'s docstring) into `CMAEvolutionStrategy.minimize`'s keyword arguments.
Split out of `optimize!` so the CMA-ES-specific option handling (scalar-vs-vector
bounds, the `sigma0` default, forwarding `seed` only when given) doesn't clutter
the method-dispatch branch above.

IPOP-CMA-ES-style restarts (Auger & Hansen, 2005): each of `:restarts` additional
runs beyond the first multiplies the population size (`:incpopsize`, default 2x)
from the previous run's, giving later restarts more diversity/global search
power. `CMAEvolutionStrategy.minimize` often converges - via its own internal
TolFun/TolX checks - well before exhausting the evaluation budget it's given;
restarting with a bigger population instead of just accepting an early
convergence spends that otherwise-unused budget on a better chance to escape
whatever local optimum the smaller population settled into, rather than being
wasted. Each restart gets an even share of `:maxfevals`, starts from the same
`x0` (never a random point or the previous restart's incumbent - a caller who
already has a good starting guess loses nothing, matching
`nelder_mead_optimize`'s restarts), and the best result across every restart
wins - restarting can only find something at least as good as the very first
run, never worse.
"""
function cma_es_optimize(f, x0::Array{Float64, 1}, options::Dict{Symbol, Any})
    n = length(x0)
    lower = get(options, :lower, 0.0)
    upper = get(options, :upper, 5000.0)
    lower_bounds = lower isa AbstractVector ? convert(Array{Float64, 1}, lower) : fill(convert(Float64, lower), n)
    upper_bounds = upper isa AbstractVector ? convert(Array{Float64, 1}, upper) : fill(convert(Float64, upper), n)

    sigma0 = get(options, :sigma0, (upper_bounds[1] - lower_bounds[1]) / 4)
    maxfevals = get(options, :maxfevals, 15000)
    restarts = get(options, :restarts, 3)
    incpopsize = get(options, :incpopsize, 2)
    per_run_maxfevals = max(1, maxfevals ÷ (restarts + 1))

    minimize_kwargs = Dict{Symbol, Any}(
        :lower => lower_bounds,
        :upper => upper_bounds,
        :maxfevals => per_run_maxfevals,
        :verbosity => get(options, :verbosity, 0),
    )
    haskey(options, :seed) && (minimize_kwargs[:seed] = options[:seed])

    popsize = get(options, :popsize, CMAEvolutionStrategy.default_popsize(n))

    result = CMAEvolutionStrategy.minimize(f, x0, sigma0; minimize_kwargs..., popsize=popsize)
    best_x = CMAEvolutionStrategy.xbest(result)
    best_f = CMAEvolutionStrategy.fbest(result)

    for _ in 1:restarts
        popsize *= incpopsize
        result = CMAEvolutionStrategy.minimize(f, x0, sigma0; minimize_kwargs..., popsize=popsize)
        candidate_f = CMAEvolutionStrategy.fbest(result)
        if candidate_f < best_f
            best_f = candidate_f
            best_x = CMAEvolutionStrategy.xbest(result)
        end
    end

    return best_x
end

"""
    nelder_mead_optimize(f, x0, options)

Minimizes `f` with iterated-local-search-style multi-start Nelder-Mead (Optim.jl),
translating this package's `optimize!(...; method=:nelder_mead, nelder_mead_options=...)`
options (see `optimize!`'s docstring). One run is seeded at `x0`; each of
`options[:restarts]` more perturbs the *current incumbent best* by a random step
scaled to `options[:restart_scale]` times the box's range in each dimension, rather
than drawing a fresh uniformly random point from the whole box. Each run gets an
even share of `:maxfevals`; the best result across all runs wins.

Earlier versions of this function drew every restart uniformly from the whole
`[lower, upper]` box. benchmark/compare_optimizers.jl's beer_game comparison (6
parameters, vs. newsvendor's 2) showed this was actively harmful, not just
inefficient: a uniform sample of a 6-dimensional box is overwhelmingly likely to
land far from any good region, so nearly every restart wasted its whole budget
converging to a poor local optimum instead of ever improving on the x0-seeded run -
visible as wildly inconsistent, often much worse than :custom/:cma_es results.
Perturbing around the incumbent instead keeps restarts a local
refinement/escape-a-nearby-local-optimum mechanism, which does not need to get
harder as dimensionality grows the way "land anywhere useful in the full box"
does - at the cost of being less able to find a *distant* better region than a
true global restart could. `x0` itself is never perturbed away from, so a caller
that already has a good starting guess loses nothing.

Candidates are clamped into `[lower, upper]` before being passed to `f` rather than
handled via Optim.jl's `Fminbox` - plain `NelderMead` needs no gradient, and Fminbox's
log-barrier approach is built around methods that have one, so a barrier around a
derivative-free simplex search isn't a well-supported combination.
"""
function nelder_mead_optimize(f, x0::Array{Float64, 1}, options::Dict{Symbol, Any})
    n = length(x0)
    lower = get(options, :lower, 0.0)
    upper = get(options, :upper, 5000.0)
    lower_bounds = lower isa AbstractVector ? convert(Array{Float64, 1}, lower) : fill(convert(Float64, lower), n)
    upper_bounds = upper isa AbstractVector ? convert(Array{Float64, 1}, upper) : fill(convert(Float64, upper), n)
    box_range = upper_bounds .- lower_bounds

    maxfevals = get(options, :maxfevals, 15000)
    restarts = get(options, :restarts, 5)
    restart_scale = get(options, :restart_scale, 0.2)

    clamped_f = x -> f(clamp.(x, lower_bounds, upper_bounds))
    per_start_budget = max(1, maxfevals ÷ (restarts + 1))

    # iterations is capped at the same budget as f_calls_limit - Options' own
    # default (1_000) would otherwise silently cut a run short before
    # f_calls_limit binds, for any per_start_budget above that.
    nm_options = Optim.Options(f_calls_limit=per_start_budget, iterations=per_start_budget)

    best_x = clamp.(x0, lower_bounds, upper_bounds)
    best_f = clamped_f(best_x)

    result = Optim.optimize(clamped_f, best_x, Optim.NelderMead(), nm_options)
    if Optim.minimum(result) < best_f
        best_f = Optim.minimum(result)
        best_x = clamp.(Optim.minimizer(result), lower_bounds, upper_bounds)
    end

    for _ in 1:restarts
        start = clamp.(best_x .+ restart_scale .* box_range .* (2 .* rand(n) .- 1), lower_bounds, upper_bounds)
        result = Optim.optimize(clamped_f, start, Optim.NelderMead(), nm_options)
        candidate_f = Optim.minimum(result)
        if candidate_f < best_f
            best_f = candidate_f
            best_x = clamp.(Optim.minimizer(result), lower_bounds, upper_bounds)
        end
    end

    return best_x
end

"""
    bayesopt_optimize(f, x0, options)

Minimizes `f` via Bayesian optimization (Surrogates.jl's Kriging surrogate,
optimized with Expected Improvement), translating this package's
`optimize!(...; method=:bayesopt, bayesopt_options=...)` options (see
`optimize!`'s docstring) into Surrogates.jl's API.

An earlier version of this function used BayesianOptimization.jl with a
GaussianProcesses.jl backend, but BayesianOptimization.jl's registered
releases only support SpecialFunctions versions incompatible with this
package's other dependencies (CMAEvolutionStrategy/Distributions already
require a modern SpecialFunctions) - an unsatisfiable Pkg.resolve() the first
time this was actually tried in CI, not something visible from reading either
package's docs. Surrogates.jl (part of the actively-maintained SciML
ecosystem) has no SpecialFunctions dependency at all, so it doesn't hit the
same conflict.

Unlike `:custom`/`:cma_es`/`:nelder_mead` - all of which spend their entire
evaluation budget on real (expensive) calls to `f` - Bayesian optimization
fits a cheap surrogate model of `f` from the points observed so far and uses
it to pick each next point to actually evaluate, trading surrogate-fitting
cost for far fewer real evaluations. That trade only pays off when real
evaluations are expensive relative to the surrogate (true here: `f` is a full
discrete-event simulation) and the parameter count is small (the surrogate
model's own cost grows with both dimensions and points observed) - both hold
for the problems this package has been benchmarked against so far (2-6
parameters), but neither is guaranteed for every caller, which is why this
method exists alongside the other three rather than replacing any of them.

`x0` isn't used as a search seed the way the other methods use it: the
initial Kriging surrogate is always built from its own space-filling (Sobol
sequence) design over `[lower, upper]`, so a caller's starting guess doesn't
bias where that initial exploration lands - the same reasoning the previous
BayesianOptimization.jl version's docstring gave, still true here.

Surrogates.jl represents a point as a plain `Float64` when there's only one
parameter, but as a `Tuple` once there's more than one (see its own test
suite) - `point_to_vec` bridges either representation back to the plain
`Vector{Float64}` `f` expects.

Returns the best point Surrogates.jl actually observed (`argmin` over the
surrogate's own recorded `x`/`y`), the same "trust only what was actually
measured" principle `:custom`/`:cma_es`/`:nelder_mead` all follow by
construction.
"""
function bayesopt_optimize(f, x0::Array{Float64, 1}, options::Dict{Symbol, Any})
    n = length(x0)
    lower = get(options, :lower, 0.0)
    upper = get(options, :upper, 5000.0)
    lower_bounds = lower isa AbstractVector ? convert(Array{Float64, 1}, lower) : fill(convert(Float64, lower), n)
    upper_bounds = upper isa AbstractVector ? convert(Array{Float64, 1}, upper) : fill(convert(Float64, upper), n)

    maxfevals = get(options, :maxfevals, 200)
    initializer_iterations = get(options, :initializer_iterations, min(5 * n, maxfevals ÷ 2))
    maxiters = max(1, maxfevals - initializer_iterations)
    num_new_samples = get(options, :num_new_samples, 100)

    point_to_vec(p) = n == 1 ? [convert(Float64, p)] : convert(Array{Float64, 1}, collect(p))
    surrogate_f = p -> f(point_to_vec(p))

    lb = n == 1 ? lower_bounds[1] : lower_bounds
    ub = n == 1 ? upper_bounds[1] : upper_bounds

    xs = Surrogates.sample(initializer_iterations, lb, ub, Surrogates.SobolSample())
    ys = surrogate_f.(xs)
    surrogate = Surrogates.Kriging(xs, ys, lb, ub)

    Surrogates.surrogate_optimize!(surrogate_f, Surrogates.EI(), lb, ub, surrogate,
                                    Surrogates.SobolSample();
                                    maxiters = maxiters, num_new_samples = num_new_samples)

    _, index = findmin(surrogate.y)
    return clamp.(point_to_vec(surrogate.x[index]), lower_bounds, upper_bounds)
end

function bboptimize(f, x0, params)
    start = Dates.now()
    latest = start

    best_f = f(x0)
    best_x = copy(x0)
    
    last_progress = 0

    pool_size = 6
    candidate_pool = [rand(length(x0)) .* (params[:SearchRange][2] - params[:SearchRange][1]) .+ params[:SearchRange][1] for i in 1:pool_size]
    #println(candidate_pool)
    pool_f = [f(candidate) for candidate in candidate_pool]
    #println(pool_f)

    t = max(0.1, min(0.9, 6 / length(x0)))
    #println(t)

    for i in 1:params[:MaxFuncEvals]
        if i > last_progress + params[:MaxStepsWithoutProgress]
            println("$i, $(Dates.now() - start), $best_f, $best_x")
            break
        end

        i1 = rand(1:pool_size)
        i2 = rand(1:pool_size)
        i3 = rand(1:pool_size)

        candidate = copy(candidate_pool[i1])
        @inbounds for j in eachindex(candidate)
            r = rand()
            if r < 0.01
                candidate[j] = params[:SearchRange][1]
            elseif r < 0.02
                candidate[j] = candidate_pool[i1][j] + 2 * (randn())
            elseif r < 0.03
                k = rand(1:length(candidate))
                candidate[j] = candidate_pool[i1][k] + 1 * (randn())
                candidate[k] = candidate_pool[i1][j] + 1 * (randn())
            elseif r < 0.12
                candidate[j] = candidate_pool[i2][j] + 1 * (randn())
            elseif r < t + 0.12
                candidate[j] = candidate_pool[i1][j] + (rand() + 0.3) * (best_x[j] - candidate_pool[i3][j]) + (randn()) / 2
            end
            if candidate[j] < params[:SearchRange][1]
                candidate[j] = params[:SearchRange][1] + rand()^3 * (params[:SearchRange][2] - params[:SearchRange][1])
            end
            if candidate[j] > params[:SearchRange][2]
                candidate[j] = params[:SearchRange][2] - rand()^3 * (params[:SearchRange][2] - params[:SearchRange][1])
            end
        end
        candidate_f = f(candidate)
        #println("$i, $(Dates.now() - start), $candidate_f, $candidate")
        if (candidate_f ≈ pool_f[i1]) && (sum(candidate_f) < sum(pool_f[i1]))
            pool_f[i1] = candidate_f
            candidate_pool[i1] = candidate
        end
        if candidate_f < pool_f[i1]
            pool_f[i1] = candidate_f
            candidate_pool[i1] = candidate
        end
        if (candidate_f ≈ best_f) && (sum(candidate_f) < sum(best_f))
            best_f = candidate_f
            best_x = copy(candidate)
            println("*- $i, $(Dates.now() - start), $best_f, $best_x")
        end
        if candidate_f < best_f
            best_f = candidate_f
            best_x = copy(candidate)
            last_progress = i
            println("** $i, $(Dates.now() - start), $best_f, $best_x")
        end

        if i % 200 == 0
            println("$i, $(Dates.now() - start), $(Dates.now() - latest), $best_f")#, $best_x")
            latest = Dates.now()
        end
    end
    return best_x
end

#function minimizer(results::BlackBoxOptim.OptimizationResults)
#    return best_candidate(results)
#end

function minimizer(x)
    return x
end