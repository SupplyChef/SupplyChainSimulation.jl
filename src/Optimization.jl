
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

    `customer_backlog` (default `false`, matching every trial simulation's
    behavior before this field existed) is passed straight through to `Env` -
    see its docstring there for what it changes.

    `params` is typed `Dict{Symbol}` (any value type), not `Dict{Symbol,
    Float64}`: the internal defaults it gets merged with already mix value
    types (`:SearchRange => (-0.0, 5000.0)` is a `Tuple`, `:NumDimensions` an
    `Int`), so a caller overriding `:SearchRange` - e.g. to narrow the search
    around a known-reasonable scale - was never actually representable as a
    `Dict{Symbol, Float64}` in the first place. `:SearchRange` itself may
    also be a `Vector` of per-dimension `(lo, hi)` tuples instead of one
    shared tuple - see `_search_bounds`'s docstring below.

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

    `custom_options` (only used when `method === :custom`) accepts:
    - `:surrogate_seed_samples`: if greater than `0`, spends that many real
      evaluations up front fitting `quadratic_surrogate_optimum` (a quadratic
      response surface over `params[:SearchRange]`, see its docstring) and
      seeds one member of `bboptimize`'s initial population with that
      surrogate's predicted optimum instead of a purely random point.
      Default `0` (disabled) - every existing caller sees byte-for-byte
      identical behavior, since this option didn't exist before.

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
function optimize!(lane_policies, supplychains...; params::Dict{Symbol}=Dict{Symbol, Float64}(), cost_function=s->-get_total_sales(s) + get_total_lost_sales(s) + get_total_holding_costs(s) + get_total_trip_fixed_costs(s) + get_total_trip_unit_costs(s) + 0.001 * get_total_orders(s), record_history::Bool=true, customer_backlog::Bool=false, method::Symbol=:custom, cma_es_options::Dict{Symbol, Any}=Dict{Symbol, Any}(), nelder_mead_options::Dict{Symbol, Any}=Dict{Symbol, Any}(), bayesopt_options::Dict{Symbol, Any}=Dict{Symbol, Any}(), custom_options::Dict{Symbol, Any}=Dict{Symbol, Any}())
    initial_states = State.(supplychains)
    envs = [Env(supplychain, initial_states, lane_policies; record_history=record_history, customer_backlog=customer_backlog) for supplychain in supplychains]

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

    x0 = vcat([get_parameters(policy) for policy in policies]...)
    x0 = convert(Array{Float64, 1}, x0)

    f = x -> minimize!(lane_policies, policies, collect(envs), collect(initial_states), x; cost_function=cost_function)

    if method === :custom
        custom_params = merge(Dict(:MaxFuncEvals => 15000,
                             :MaxStepsWithoutProgress => 1500,
                             :SearchRange => (-0.0, 5000.0),
                             :NumDimensions => length(x0)), params)

        # Disabled (0) unless a caller explicitly opts in via custom_options -
        # every existing caller (this option didn't exist before) sees
        # seed_candidates stay `nothing`, i.e. byte-for-byte identical
        # behavior to before this option existed. See
        # quadratic_surrogate_optimum's docstring for what this buys: one
        # gradient-informed population member instead of purely random ones,
        # validated like any other candidate (never trusted blindly).
        surrogate_seed_samples = get(custom_options, :surrogate_seed_samples, 0)
        seed_candidates = nothing
        if surrogate_seed_samples > 0
            lower, upper = custom_params[:SearchRange]
            seed = quadratic_surrogate_optimum(f, fill(convert(Float64, lower), length(x0)), fill(convert(Float64, upper), length(x0)); samples=surrogate_seed_samples)
            seed_candidates = [seed]
            # Comes out of the same :MaxFuncEvals budget, not on top of it -
            # a fair comparison against plain :custom at the same total
            # evaluation count, not an unadvertised extra allowance.
            custom_params[:MaxFuncEvals] = max(1, custom_params[:MaxFuncEvals] - surrogate_seed_samples)
        end

        best = SupplyChainSimulation.bboptimize(f, x0, custom_params; seed_candidates=seed_candidates, verbose=get(custom_options, :verbose, false))
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
    sobol_indices(f, lower, upper; samples)

Computes first-order (`S1`) and total-order (`ST`) Sobol sensitivity indices
of `f` over the box `lower[i] <= x[i] <= upper[i]`, via the Saltelli
(2010)/Jansen (1999) estimators - the same estimators a SciML-ecosystem
sensitivity-analysis package (GlobalSensitivity.jl) would use, reimplemented
directly here rather than depending on that package: GlobalSensitivity.jl
bundles many sensitivity methods together, and two of its own transitive
dependencies (Copulas.jl, needed only for its unrelated Shapley-effects
method, and ForwardDiff) turned out to require mutually incompatible
versions of GlobalSensitivity.jl itself once resolved against this
package's other, already-modern dependencies - not something fixable by
adjusting a compat bound on our end. Kept as a standalone, generically
useful function (not folded into `sensitivity_analysis` below) specifically
so a real SciML package could be swapped in later as a pure implementation
detail, without `sensitivity_analysis`'s own signature or return shape
needing to change, if a working dependency path for one ever exists.

Draws two independent `samples x length(lower)` matrices `A`/`B` uniformly
from the box, plus one `A` with column `i` swapped in from `B` per parameter
`i` - `samples * (length(lower) + 2)` evaluations of `f` in total. `S1`/`ST`
are validated in this package's tests against the analytically known Sobol
indices of a simple linear function (which has no parameter interactions,
so `S1 == ST` exactly in the infinite-sample limit) rather than only
smoke-tested, since there is no local Julia available in this project's
usual workflow to cross-check the estimator against a reference
implementation interactively.
"""
function sobol_indices(f, lower::Array{Float64, 1}, upper::Array{Float64, 1}; samples::Int)
    d = length(lower)
    box_range = upper .- lower

    A = [lower .+ rand(d) .* box_range for _ in 1:samples]
    B = [lower .+ rand(d) .* box_range for _ in 1:samples]

    f_A = f.(A)
    f_B = f.(B)

    all_f = vcat(f_A, f_B)
    mean_f = sum(all_f) / length(all_f)
    var_y = sum((y - mean_f)^2 for y in all_f) / (length(all_f) - 1)

    S1 = zeros(d)
    ST = zeros(d)
    for i in 1:d
        AB_i = [copy(a) for a in A]
        for j in 1:samples
            AB_i[j][i] = B[j][i]
        end
        f_ABi = f.(AB_i)

        S1[i] = (sum(f_B[j] * (f_ABi[j] - f_A[j]) for j in 1:samples) / samples) / var_y
        ST[i] = (sum((f_A[j] - f_ABi[j])^2 for j in 1:samples) / (2 * samples)) / var_y
    end

    return S1, ST
end

"""
    sensitivity_analysis(lane_policies, supplychains...; cost_function, record_history, options)

Runs a global (Sobol) sensitivity analysis of `cost_function` against every
policy parameter across `lane_policies`, via `sobol_indices` above - a
different question from `optimize!`'s "what's the best policy?": "which
policy parameters actually matter for cost?" Useful as a triage step before
spending an `optimize!` budget tuning parameters that barely move the
outcome, or for understanding *why* a policy landed where it did.

Shares `optimize!`'s setup (Env construction, sorted policy keys, x0
assembly, the same `minimize!`-based evaluator - see `optimize!`'s comment
on why `sorted_keys`' order has to be deterministic) rather than reimplementing
it, so the parameter ordering `S1`/`ST` are reported in exactly matches what
`optimize!` itself searches over.

Restores every policy's original parameters before returning: `minimize!`
mutates each policy's parameters as a side effect of evaluating a candidate
`x` (see its definition above) - without explicitly restoring `x0` afterward,
this function, despite being a read-only diagnostic, would otherwise leave
every policy set to whatever the last of `sobol_indices`' many sampled
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
- `:samples`: controls evaluation cost - needs `samples * (length(x0) + 2)`
  real evaluations (see `sobol_indices`). Default `1000`.
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

    S1, ST = sobol_indices(f, lower_bounds, upper_bounds; samples=samples)

    i = 1
    for policy in policies
        set_parameters!(policy, x0[i:i+length(get_parameters(policy))-1])
        i = i + length(get_parameters(policy))
    end

    return (parameter_labels = parameter_labels, S1 = S1, ST = ST)
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

Default `:upper` is `200.0`, not the `5000.0` every other `*_optimize`
function here defaults to: `benchmark/compare_optimizers.jl`'s first real
head-to-head comparison (5 trials x 3000 evals) showed `:cma_es` regressing
badly against `:custom` on the 6-parameter beer_game problem (positive/bad
training cost vs. `:custom`'s negative/good) while tied on the 2-parameter
newsvendor problem - traced to `sigma0`'s default of `(upper-lower)/4`,
which comes out to `1250` against the shared `5000.0` box, an enormous step
size relative to where these policies' real parameters live (order-up-to/
coverage levels in the tens to low hundreds, confirmed by both the
sensitivity-report smoke tests' bounds and this fix's own validation run).
Re-running the same comparison with `upper=200.0` (`sigma0=50`) turned
`:cma_es` into the best or tied-best method on both problems - unlike
`:custom`'s search dynamics (population-based, selection-driven, tolerant of
an oversized box) or `nelder_mead_optimize`'s (anchored to the incumbent,
not the raw box), CMA-ES's sampling distribution is directly sized by
`sigma0`, so an oversized box translates directly into an oversized initial
search radius. Still fully overridable via `:upper`/`:sigma0` for callers
whose real parameter scale genuinely is in the thousands.
"""
function cma_es_optimize(f, x0::Array{Float64, 1}, options::Dict{Symbol, Any})
    n = length(x0)
    lower = get(options, :lower, 0.0)
    upper = get(options, :upper, 200.0)
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

Returns the best point actually observed (`argmin` over every point measured,
across both the initial design and the EI loop), the same "trust only what
was actually measured" principle `:custom`/`:cma_es`/`:nelder_mead` all
follow by construction.

Default `:upper` is `200.0`, not `5000.0` - see `cma_es_optimize`'s docstring
for the shared reasoning (the same `benchmark/compare_optimizers.jl` run that
found `:cma_es` regressing on beer_game found `:bayesopt` far worse: 300x
`:custom`'s cost on newsvendor, roughly 10 orders of magnitude worse on
beer_game).

Unlike `:cma_es`, tightening the box only closed part of `:bayesopt`'s gap
(newsvendor: 2.79e6 down to 8.7e4, still ~10x `:custom`'s 9.0e3; beer_game:
3.97e13 down to 2.6e9, still ~2e5x `:custom`'s -1.2e4) - both runs' own
convergence tables showed the reported cost identical from 10% to 100% of
the evaluation budget, meaning the search stopped improving almost
immediately regardless of box size. Reading Surrogates.jl's own
`surrogate_optimize!(..., EI(), ...)` source (this function no longer calls
it - see below) turned up two compounding causes, neither fixed by a tighter
box alone:

1. The Kriging surrogate's `theta` (length-scale) and `p` (smoothness,
   defaulting to `2.0` - i.e. assumes an infinitely smooth function) are
   estimated once at construction and never re-estimated: `update!` (the
   function `surrogate_optimize!` uses to add each new point) explicitly
   reuses the same `theta`/`p` forever. This package's cost functions are
   the opposite of smooth - `ceil(Int, deficit)` order-quantity truncation
   (see `BackwardCoverageOrderingPolicy`/`ForwardCoverageOrderingPolicy` in
   Policy.jl) and inventory clamps make them full of kinks - so a "this
   looks smooth" fit from the first ~30 points stays wrong for the rest of
   the run instead of correcting itself as evidence accumulates.
2. `surrogate_optimize!`'s early-termination check is `new_EI_max < 1e-6 *
   (maximum(krig.y) - minimum(krig.y))` - relative to the observed y-range,
   not an absolute tolerance. In 6 dimensions, even a `[0, 200]` box lets a
   handful of the initial Sobol samples land in badly-overordered territory
   with catastrophic cost, which alone inflates that range into the millions
   - which inflates the absolute EI the search needs to clear to keep going
   into a threshold no genuine improvement can meet. This is worse for
   beer_game (6 dimensions, more outlier-prone) than newsvendor (2), matching
   the gap each still had after the box fix.

Both point to the same underlying issue: Surrogates.jl's basic `Kriging` +
`EI()` recipe, used as-is, isn't built for a highly non-smooth,
wide-dynamic-range objective. Rather than a tighter box, this function
reimplements `surrogate_optimize!`'s EI loop directly (same acquisition
function, verified against Surrogates.jl's own source) with two changes:
`theta`/`p` are re-estimated by rebuilding the Kriging surrogate from every
point observed so far (not just appending via `update!`) on a geometric
schedule (whenever the point count has at least doubled since the last
rebuild - bounding the number of expensive full refits to O(log(maxfevals))
the same way IPOP-CMA-ES's restarts bound *their* growth, rather than paying
a full refit's O(points^3) cost every single iteration), and the internal
early-termination check is dropped in favor of an explicit `:ei_iterations`
cap (default `200`, matching this function's own default `:maxfevals`): a
first version of this fix dropped the early-termination check without
replacing the runtime bound it happened to also provide, and a
`benchmark/compare_optimizers.jl` run (which forces `:maxfevals` up to 3000
to match the other three methods) ran for 30+ minutes and had to be
cancelled - `num_new_samples` (default 100) candidates scored against a
growing Kriging surrogate on every one of ~3000 iterations is expensive
regardless of how correct the acquisition function is. Past `:ei_iterations`
real evaluations, this function switches to plain space-filling samples
evaluated directly with no surrogate/EI cost - still real, budget-spending
evaluations, just without per-point acquisition cost whose marginal value
is small once the surrogate already has hundreds of points.
"""
function bayesopt_optimize(f, x0::Array{Float64, 1}, options::Dict{Symbol, Any})
    n = length(x0)
    lower = get(options, :lower, 0.0)
    upper = get(options, :upper, 200.0)
    lower_bounds = lower isa AbstractVector ? convert(Array{Float64, 1}, lower) : fill(convert(Float64, lower), n)
    upper_bounds = upper isa AbstractVector ? convert(Array{Float64, 1}, upper) : fill(convert(Float64, upper), n)

    maxfevals = get(options, :maxfevals, 200)
    initializer_iterations = get(options, :initializer_iterations, min(5 * n, maxfevals ÷ 2))
    maxiters = max(1, maxfevals - initializer_iterations)
    num_new_samples = get(options, :num_new_samples, 100)
    # EI-guided iterations are the expensive part of this function
    # (num_new_samples candidates scored against a growing Kriging surrogate
    # every iteration) - capped independent of maxfevals so a caller forcing
    # a much larger budget than bayesopt_optimize's own default
    # (benchmark/compare_optimizers.jl deliberately matches the other three
    # methods' 3000-eval budget for a fair comparison - see this function's
    # docstring) doesn't pay that cost thousands of times over: a first
    # attempt without this cap ran for 30+ minutes and had to be cancelled.
    # Any budget beyond ei_iterations is spent on plain space-filling
    # samples instead - still real, informative evaluations, just without
    # per-point EI scoring, whose marginal value is small once the
    # surrogate already has hundreds of points. Default 200 matches this
    # function's own default :maxfevals, so ordinary (non-forced-budget)
    # callers see unchanged behavior - the cap only binds when maxfevals is
    # pushed well past that.
    ei_iterations = min(maxiters, get(options, :ei_iterations, 200))

    point_to_vec(p) = n == 1 ? [convert(Float64, p)] : convert(Array{Float64, 1}, collect(p))
    surrogate_f = p -> f(point_to_vec(p))

    lb = n == 1 ? lower_bounds[1] : lower_bounds
    ub = n == 1 ? upper_bounds[1] : upper_bounds

    xs = Surrogates.sample(initializer_iterations, lb, ub, Surrogates.SobolSample())
    ys = surrogate_f.(xs)
    surrogate = Surrogates.Kriging(collect(xs), collect(ys), lb, ub)

    point_distance(a, b) = n == 1 ? abs(a - b) : sqrt(sum((collect(a) .- collect(b)) .^ 2))
    dtol = 1.0e-3 * point_distance(lb, ub)

    # EI acquisition, matching Surrogates.jl's own EI() formula exactly (see
    # this function's docstring) - reimplemented directly so the refit
    # schedule below can replace its internal update!-only/early-stopping
    # loop.
    eps = 0.01
    next_refit_at = 2 * length(surrogate.x)
    for i in 1:ei_iterations
        candidates = collect(Surrogates.sample(num_new_samples, lb, ub, Surrogates.SobolSample()))
        f_min = minimum(surrogate.y)

        evaluations = Vector{Float64}(undef, length(candidates))
        for (j, candidate) in enumerate(candidates)
            std = Surrogates.std_error_at_point(surrogate, candidate)
            u = surrogate(candidate)
            z = abs(std) > 1.0e-6 ? (f_min - u - eps) / std : 0.0
            evaluations[j] = (f_min - u - eps) * Distributions.cdf(Distributions.Normal(), z) +
                              std * Distributions.pdf(Distributions.Normal(), z)
        end

        # Surrogates.sample(..., SobolSample()) is a deterministic
        # low-discrepancy sequence - a fresh call with the same
        # num_new_samples/lb/ub returns the *same* points every iteration,
        # so the EI argmax can (and, empirically, does) re-pick a point
        # already in surrogate.x. Kriging's correlation matrix is singular
        # for duplicate/near-duplicate x's, and Surrogates.Kriging doesn't
        # throw on that - it prints "cannot build Kriging" and returns
        # `nothing`, which then crashes on the next access. Skip any
        # candidate within dtol of an existing point (Surrogates.jl's own
        # surrogate_optimize! does the same dedup, with the same dtol
        # formula) and fall back to a genuinely non-deterministic draw if
        # every candidate this round turns out to be a duplicate.
        best_candidate = nothing
        while !isempty(candidates)
            index_max = argmax(evaluations)
            candidate = candidates[index_max]
            if any(point_distance(candidate, x) < dtol for x in surrogate.x)
                deleteat!(candidates, index_max)
                deleteat!(evaluations, index_max)
            else
                best_candidate = candidate
                break
            end
        end
        if best_candidate === nothing
            best_candidate = first(Surrogates.sample(1, lb, ub, Surrogates.RandomSample()))
        end

        y_new = surrogate_f(best_candidate)

        if length(surrogate.x) + 1 >= next_refit_at || i == ei_iterations
            surrogate = Surrogates.Kriging(vcat(surrogate.x, [best_candidate]), vcat(surrogate.y, [y_new]), lb, ub)
            next_refit_at = 2 * length(surrogate.x)
        else
            Surrogates.update!(surrogate, best_candidate, y_new)
        end
    end

    _, index = findmin(surrogate.y)
    best_x = surrogate.x[index]
    best_y = surrogate.y[index]

    # Any budget left after ei_iterations (see above) - plain space-filling
    # samples evaluated directly, with no surrogate/EI cost at all, since
    # the surrogate isn't refit again before returning anyway.
    for _ in 1:(maxiters - ei_iterations)
        candidate = first(Surrogates.sample(1, lb, ub, Surrogates.RandomSample()))
        y = surrogate_f(candidate)
        if y < best_y
            best_y = y
            best_x = candidate
        end
    end

    return clamp.(point_to_vec(best_x), lower_bounds, upper_bounds)
end

"""
    quadratic_surrogate_optimum(f, lower, upper; samples)

Fits a quadratic response surface `y ≈ c0 + b'x + x'Ax` to `samples` real
evaluations of `f` (a space-filling Sobol design over the box `[lower,
upper]`), via ordinary least squares, then returns *that surrogate's own*
closed-form predicted optimum (solving `∇y = b + 2Ax = 0`, i.e. `x* = -A \\
b`), clamped into `[lower, upper]`.

Built as a small, standalone alternative to `:bayesopt`'s Kriging surrogate
(see `bayesopt_optimize`'s docstring for the two compounding bugs that
approach hit on this package's non-smooth cost functions: stale
hyperparameters, and an early-termination check sensitive to a few
catastrophic-cost outliers) - a quadratic response surface has no
hyperparameters to estimate at all, and its gradient/optimum are exact
closed-form linear algebra rather than an iterative fit, so neither failure
mode applies. The tradeoff is a much cruder model: a single global quadratic
can't capture multiple local optima or genuinely non-quadratic structure the
way a Kriging surrogate (in principle) can - this is meant to seed one
promising candidate for another search to actually validate against the real
`f`, not to be trusted as a standalone optimizer.

Requires `samples` to be at least the number of quadratic-model coefficients
(`1 + 2n + n(n-1)/2` for `n = length(lower)` parameters: one intercept, `n`
linear terms, `n` squared terms, `n(n-1)/2` cross terms) - fewer than that
makes `X \\ ys` an underdetermined least-squares solve with no real
predictive meaning. Falls back to the best point actually observed among the
sampled evaluations (rather than trusting an ill-posed solve) if `A` turns
out non-invertible.
"""
function quadratic_surrogate_optimum(f, lower::Array{Float64, 1}, upper::Array{Float64, 1}; samples::Int)
    n = length(lower)
    nfeatures = 1 + 2 * n + (n * (n - 1)) ÷ 2

    xs = Surrogates.sample(samples, lower, upper, Surrogates.SobolSample())
    xs_vec = [n == 1 ? [convert(Float64, x)] : convert(Array{Float64, 1}, collect(x)) for x in xs]
    ys = [convert(Float64, f(x)) for x in xs_vec]

    X = zeros(samples, nfeatures)
    for (row, x) in enumerate(xs_vec)
        col = 1
        X[row, col] = 1.0
        col += 1
        for i in 1:n
            X[row, col] = x[i]
            col += 1
        end
        for i in 1:n
            X[row, col] = x[i]^2
            col += 1
        end
        for i in 1:n, j in (i + 1):n
            X[row, col] = x[i] * x[j]
            col += 1
        end
    end

    coeffs = X \ ys
    b = coeffs[2:(1 + n)]

    A = zeros(n, n)
    for i in 1:n
        A[i, i] = coeffs[1 + n + i]
    end
    idx = 1 + 2 * n
    for i in 1:n, j in (i + 1):n
        idx += 1
        A[i, j] = coeffs[idx] / 2
        A[j, i] = coeffs[idx] / 2
    end

    candidate = try
        x_star = -(2 .* A) \ b
        if all(isfinite, x_star)
            clamp.(x_star, lower, upper)
        else
            xs_vec[argmin(ys)]
        end
    catch e
        e isa Union{LinearAlgebra.SingularException, LinearAlgebra.LAPACKException} || rethrow()
        xs_vec[argmin(ys)]
    end

    return candidate
end

"""
    _search_bounds(params, j)::Tuple{Float64,Float64}

`params[:SearchRange]` is either a single `(lo, hi)` tuple, applied to every
dimension alike (the original, still-default behavior), or a `Vector` of
`(lo, hi)` tuples, one per dimension - e.g. so a dimensionless gain
parameter and an inventory-level parameter can get genuinely different
exploration ranges instead of being forced to share one box sized for
whichever needs to be bigger. This is the single indirection every
`params[:SearchRange][1|2]` use below goes through, so both forms work
without duplicating the branch at each call site.
"""
_search_bounds(params, j) = params[:SearchRange] isa AbstractVector ? params[:SearchRange][j] : params[:SearchRange]

"""
    bboptimize(f, x0, params; seed_candidates=nothing, verbose=false)

`seed_candidates` (default `nothing`, matching every existing caller's
behavior byte-for-byte) optionally replaces the first `length(seed_candidates)`
members of the initial random `candidate_pool` with given points instead -
e.g. `quadratic_surrogate_optimum`'s predicted optimum, giving the
population a genuinely gradient-informed starting guess alongside its
otherwise-random members. Seeded candidates are still just population
members like any other: nothing about the search treats them specially past
initialization, so a bad seed costs nothing beyond one wasted pool slot.

`verbose` (default `false`) controls whether progress is printed to stdout
- matching `cma_es_options[:verbosity]`'s default-off convention for the
other optimizer methods, so calling `optimize!()` doesn't print by default.
"""
function bboptimize(f, x0, params; seed_candidates::Union{Nothing, Vector{Vector{Float64}}}=nothing, verbose::Bool=false)
    start = Dates.now()
    latest = start

    best_f = f(x0)
    best_x = copy(x0)

    last_progress = 0

    pool_size = 6
    candidate_pool = [[(b = _search_bounds(params, j); b[1] + rand() * (b[2] - b[1])) for j in 1:length(x0)] for i in 1:pool_size]
    if seed_candidates !== nothing
        for (idx, seed) in enumerate(seed_candidates)
            idx > pool_size && break
            candidate_pool[idx] = [clamp(seed[j], _search_bounds(params, j)...) for j in eachindex(seed)]
        end
    end
    pool_f = [f(candidate) for candidate in candidate_pool]

    t = max(0.1, min(0.9, 6 / length(x0)))

    for i in 1:params[:MaxFuncEvals]
        if i > last_progress + params[:MaxStepsWithoutProgress]
            verbose && println("$i, $(Dates.now() - start), $best_f, $best_x")
            break
        end

        i1 = rand(1:pool_size)
        i2 = rand(1:pool_size)
        i3 = rand(1:pool_size)

        candidate = copy(candidate_pool[i1])
        @inbounds for j in eachindex(candidate)
            bounds = _search_bounds(params, j)
            r = rand()
            if r < 0.01
                candidate[j] = bounds[1]
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
            if candidate[j] < bounds[1]
                candidate[j] = bounds[1] + rand()^3 * (bounds[2] - bounds[1])
            end
            if candidate[j] > bounds[2]
                candidate[j] = bounds[2] - rand()^3 * (bounds[2] - bounds[1])
            end
        end
        candidate_f = f(candidate)
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
            verbose && println("*- $i, $(Dates.now() - start), $best_f, $best_x")
        end
        if candidate_f < best_f
            best_f = candidate_f
            best_x = copy(candidate)
            last_progress = i
            verbose && println("** $i, $(Dates.now() - start), $best_f, $best_x")
        end

        if verbose && i % 200 == 0
            println("$i, $(Dates.now() - start), $(Dates.now() - latest), $best_f")
            latest = Dates.now()
        end
    end
    return best_x
end

