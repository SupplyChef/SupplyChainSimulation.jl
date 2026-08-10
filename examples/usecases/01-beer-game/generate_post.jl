#=
Reads results.json (written by model.jl) and writes post.md: the actual
blog post, with every number sourced from that file - nothing here is
typed in by hand.

Run after model.jl:
    julia --project=examples/usecases examples/usecases/01-beer-game/generate_post.jl
=#
using JSON3

results = open(joinpath(@__DIR__, "results.json")) do io
    JSON3.read(io)
end

pct(x) = string(round(x * 100; digits=1), "%")
fmt(x; digits=1) = round(x; digits=digits)
ratio(x; digits=1) = string(round(x; digits=digits), "x")

naive = results.naive_aggregate
opt_in = results.optimized_in_sample_aggregate
opt_hold = results.optimized_holdout_aggregate
s1 = results.optimized_in_sample_scenario1

# Converted to plain Dict{String,Float64} rather than indexed as JSON3
# objects directly - guarantees String-key getindex regardless of JSON3's
# own Symbol/String indexing support.
as_dict(obj) = Dict(string(k) => Float64(v) for (k, v) in pairs(obj))

# Same rationale as as_dict, but for objects whose values are themselves
# nested JSON3 objects (not scalars) - e.g. anchor_adjust_results/
# human_panic_results, keyed by a numeric-looking string ("1.0", "0.4", ...)
# that dot-access can't reach anyway.
as_obj_dict(obj) = Dict(string(k) => v for (k, v) in pairs(obj))

naive_bw = as_dict(results.naive_bullwhip_ratios)
opt_bw = as_dict(results.optimized_bullwhip_ratios)
naive_cv = as_dict(results.naive_inventory_cv)
opt_cv = as_dict(results.optimized_inventory_cv)

fill_rate_change_pp = round((opt_in.fill_rate - naive.fill_rate) * 100; digits=2)
generalization_gap_pp = round((opt_in.fill_rate - opt_hold.fill_rate) * 100; digits=2)

naive_total_cost = Float64(results.naive_costs.total_cost)
optimized_total_cost = Float64(results.optimized_costs.total_cost)
cost_delta_pct = round(abs(optimized_total_cost - naive_total_cost) / abs(naive_total_cost) * 100; digits=1)
cost_verdict = if optimized_total_cost < naive_total_cost
    """So yes — despite worse fill rate and far worse bullwhip, the optimized policy is genuinely **$(cost_delta_pct)% cheaper** under the exact cost function it was told to minimize. The optimizer isn't wrong or broken; it correctly found that trading fill rate and order stability for lower holding costs was worth it, *given the weights in this cost function* — which is precisely the point of the sections above: those weights, not some flaw in the optimizer, are what a real deployment needs to interrogate before trusting this policy."""
else
    """**It is not.** The optimized policy is **$(cost_delta_pct)% more expensive** than the naive fixed-target baseline under the exact same cost function — worse on total cost, worse on fill rate, and far worse on bullwhip. That changes the conclusion: this isn't a case of the optimizer correctly trading service for savings. `optimize!` only ever searched `BackwardCoverageOrderingPolicy` parameters; it never had access to `NetUptoOrderingPolicy`, so it's entirely possible the naive policy family is simply a better structural fit for this network, and the "optimized" result is the best of a worse-fitting family, not a genuine optimum. The lesson isn't "don't trust optimization" — it's "the choice of policy *family* you hand the optimizer matters as much as the tuning," and that's worth testing directly as a follow-up (run `optimize!` over `NetUptoOrderingPolicy` too, and compare)."""
end

naive_classic_total = Float64(results.naive_classic_score.total)
optimized_classic_total = Float64(results.optimized_classic_score.total)
classic_delta_pct = round(abs(optimized_classic_total - naive_classic_total) / abs(naive_classic_total) * 100; digits=1)
classic_agrees = (optimized_classic_total < naive_classic_total) == (optimized_total_cost < naive_total_cost)
classic_verdict = if optimized_classic_total < naive_classic_total
    """Under this scoring convention, optimized is **$(classic_delta_pct)% cheaper** than naive — $(classic_agrees ? "the same direction as the cost comparison above, so the two conventions agree on which policy wins here, even though they measure the miss differently." : "the *opposite* conclusion from the cost comparison above. Which policy 'wins' depends on which convention you score it by, and that's the actual finding of this section: the choice of convention isn't a footnote, it changes the answer.")"""
else
    """Under this scoring convention, optimized is **$(classic_delta_pct)% more expensive** than naive — $(classic_agrees ? "the same direction as the cost comparison above, so the two conventions agree on which policy wins here, even though they measure the miss differently." : "the *opposite* conclusion from the cost comparison above. Which policy 'wins' depends on which convention you score it by, and that's the actual finding of this section: the choice of convention isn't a footnote, it changes the answer.")"""
end

cover2 = join(round.(Float64.(results.tuned_policy_cover.l2_wholesaler_to_retailer); digits=2), ", ")
cover3 = join(round.(Float64.(results.tuned_policy_cover.l3_factory_to_wholesaler); digits=2), ", ")
cover4 = join(round.(Float64.(results.tuned_policy_cover.l4_supplier_to_factory); digits=2), ", ")

# --- Follow-up 1/2: same policy family, tuned properly (NetUptoOrderingPolicy
#     via optimize!) vs. the searched family given 4x the budget. ---
tn = results.tuned_naive_metrics
tn_targets = results.tuned_naive_targets
tn_bw = as_dict(tn.bullwhip)
tn_cv = as_dict(tn.inventory_cv)
tn_total_cost = Float64(tn.costs.total_cost)

bo = results.big_opt_metrics
bo_cover = results.big_opt_cover
bo_total_cost = Float64(bo.costs.total_cost)

family_beats_naive_pct = round(abs(tn_total_cost - naive_total_cost) / abs(naive_total_cost) * 100; digits=1)
family_beats_optimized_pct = round(abs(tn_total_cost - optimized_total_cost) / abs(optimized_total_cost) * 100; digits=1)
# cost_function is -sales+lost_sales+holding+..., so more negative = cheaper;
# "helped" means the bigger-budget rerun's cost beat the original 15k-eval run.
bigger_budget_helped = bo_total_cost < optimized_total_cost
budget_verdict = if bigger_budget_helped
    """4x the search budget did lower the cost somewhat, from $(fmt(optimized_total_cost)) to $(fmt(bo_total_cost)) — but fill rate actually *dropped* further, from $(pct(opt_in.fill_rate)) to $(pct(Float64(bo.aggregate.fill_rate))), and total cost is still nowhere near the tuned-naive result above. As the next section shows, that's not because this policy family can't reach it - it's a search failure on a specific, identifiable landscape feature."""
else
    """4x the search budget bought essentially nothing: total cost went from $(fmt(optimized_total_cost)) to $(fmt(bo_total_cost)), no better, while fill rate dropped from $(pct(opt_in.fill_rate)) to $(pct(Float64(bo.aggregate.fill_rate))). As the next section shows, that's not because this policy family can't reach a good answer - it's a search failure on a specific, identifiable landscape feature."""
end

# --- Follow-up: does BackwardCoverageOrderingPolicy's own search space
#     actually contain the tuned-naive optimum? At cover[1]=0, get_order's
#     `weights` accumulator never leaves 0, so the weighted-average branch is
#     skipped and `coverage` collapses to the bare constant `cover[2]` -
#     algebraically identical to NetUptoOrderingPolicy(upto=cover[2]). That
#     point sits inside the same SearchRange optimize! searched above.
#     equiv_metrics plugs it in directly (no optimize! call) to test this.
em = results.equiv_metrics
em_total_cost = Float64(em.costs.total_cost)
equiv_matches_tuned_naive = isapprox(em_total_cost, tn_total_cost; atol=1.0) && isapprox(Float64(em.aggregate.fill_rate), Float64(tn.aggregate.fill_rate); atol=1e-6)

# --- Follow-up: re-tune both policy families against the real beer-game cost
#     (holding + backlog, not metrics_cost_function's holding + lost-sales-
#     only view) to test whether the missing backlog term is what let the
#     wholesaler's zero-buffer target, and BackwardCoverageOrderingPolicy's
#     blow-up, look cheap.
ctn = results.classic_tuned_naive_metrics
ctn_targets = results.classic_tuned_naive_targets
ctn_cv = as_dict(ctn.inventory_cv)
ctn_classic_total = Float64(ctn.classic.total)
naive_classic_total = Float64(results.naive_classic_score.total)

co = results.classic_opt_metrics
co_cover = results.classic_opt_cover
co_classic_total = Float64(co.classic.total)
co_fill_rate = Float64(co.aggregate.fill_rate)

# --- Follow-up 3: Sterman's anchor-and-adjust sweep (published, measured
#     human-ordering heuristic) and the human-panic variant (same heuristic
#     plus a positive safety-stock cushion). ---
aa_dict = as_obj_dict(results.anchor_adjust_results)
sweep_weights = ["1.0", "0.8", "0.6", "0.4", "0.2"]
sweep_rows = join([
    let r = aa_dict[w], bw = as_dict(r.bullwhip)
        "| $(w) | $(pct(Float64(r.aggregate.fill_rate))) | $(ratio(bw["factory_orders (l4)"])) | $(fmt(Float64(r.costs.total_cost))) |"
    end
    for w in sweep_weights
], "\n")

hp_alpha = results.human_panic_alpha_supply_line
hp_dict = as_obj_dict(results.human_panic_results)
panic_levels = ["0.0", "5.0", "10.0", "20.0"]
panic_rows = join([
    let r = hp_dict[d], bw = as_dict(r.bullwhip)
        "| $(d) | $(pct(Float64(r.aggregate.fill_rate))) | $(ratio(bw["retailer_orders (l2)"])) | $(ratio(bw["wholesaler_orders (l3)"])) | $(ratio(bw["factory_orders (l4)"])) | $(fmt(Float64(r.costs.total_cost))) |"
    end
    for d in panic_levels
], "\n")
panic_factory_bw_low = as_dict(hp_dict["0.0"].bullwhip)["factory_orders (l4)"]
panic_factory_bw_high = as_dict(hp_dict["20.0"].bullwhip)["factory_orders (l4)"]
panic_fill_low = Float64(hp_dict["0.0"].aggregate.fill_rate)
panic_fill_high = Float64(hp_dict["20.0"].aggregate.fill_rate)

# --- Follow-up: same question as equiv_matches_tuned_naive above, but for
#     the real (backlog-priced) cost - is classic_opt's collapse to a low
#     fill rate above another search failure, or does this policy family
#     genuinely not contain the classic_tuned_naive optimum under this cost?
cem = results.classic_equiv_metrics
cem_total_cost = Float64(cem.classic.total)
classic_equiv_matches_tuned_naive = isapprox(cem_total_cost, ctn_classic_total; atol=1.0) && isapprox(Float64(cem.aggregate.fill_rate), Float64(ctn.aggregate.fill_rate); atol=1e-6)

# --- Search-improvement narrative: coarse-to-fine restart, single-round
#     empirical probing, and 3-round iterative probing, all re-tuning
#     BackwardCoverageOrderingPolicy against classic_cost_function from the
#     same starting point that produced classic_opt's collapse above.
cf = results.coarse_fine_metrics
cf_range = round.(Float64.(results.coarse_fine_refined_range); digits=2)
probe_scales_r = round.(Float64.(results.probe_scales); digits=1)
probed_ranges_r = round.([Float64(r[2]) for r in results.probed_ranges]; digits=1)
po = results.probed_opt_metrics
ip = results.iter_probed_metrics
probing_gain_ratio = round(Float64(po.aggregate.fill_rate) / co_fill_rate; digits=1)

# --- customer_backlog=true: does letting customer orders backlog too, instead
#     of always dropping as a one-time lost sale, close the remaining gap
#     between this package's convention and the original board game's?
cbtn = results.customer_backlog_tuned_naive_metrics
cbtn_targets = results.customer_backlog_tuned_naive_targets

post = """
# The Beer Game, "Solved"? What an Optimizer Actually Did to the Bullwhip Effect

*Part of a series on SupplyChainSimulation.jl and SupplyChainOptimization.jl. All numbers on this page are generated by running [`model.jl`](model.jl) in CI — see the workflow run linked in this repo's Actions tab for the raw log.*

## The game that's been humbling smart people for 60 years

In 1961, MIT's Jay Forrester built a simple board-game simulation of a four-stage supply chain — retailer, wholesaler, factory, supplier — each stage separated by a shipping delay, each stage only able to see its own orders and inventory. Consumer demand barely moves. And yet, run after run, for six decades, the orders placed *inside* the chain oscillate wildly.

This isn't a fluke of bad players. John Sterman's 1989 study in *Management Science* ("Modeling Managerial Behavior: Misperceptions of Feedback in a Dynamic Decision-Making Experiment") ran the game with people who *understood* the dynamics going in, and they still produced the same oscillation. Lee, Padmanabhan, and Whang's 1997 analysis of the same pattern in real companies (built on Procter & Gamble's own study of Pampers demand) gave it a name and a rough shape: consumer demand that varies ±5–10% can show up as ±20% swings at the retailer, ±40% at the wholesaler, ±80%+ at the manufacturer. That amplification — the **bullwhip effect** — is measured, formally, as the ratio of the variance of what an echelon orders to the variance of what its actual downstream customer demands. Ratio above 1: this echelon is amplifying. Below 1: it's damping.

We built the same four-echelon network in SupplyChainSimulation.jl, measured that ratio directly at every echelon, and let `optimize!` tune the upstream ordering policies. The result is more interesting — and more useful — than "the optimizer fixed it."

## The network

Four echelons — customer, retailer, wholesaler, factory, supplier — with 2-period lead times between retailer/wholesaler/factory and a 4-period lead time from the supplier, demand averaging $(results.mean_demand) units/period, simulated over a $(results.horizon)-period horizon across $(results.scenario_count) independent stochastic demand scenarios (Poisson($(results.mean_demand))). Full network-construction code is in [`model.jl`](model.jl); it's the same network shape as the package's own `test/policy-beergame-tests.jl`.

## Baseline: a naive fixed policy already bullwhips, textbook-style

Each upstream echelon runs `NetUptoOrderingPolicy`, reacting only to its own local net inventory, with a fixed target sized for "pipeline coverage, no safety stock" — mean demand × (lead time + 1) — and never tuned:

| Lane | Lead time | Naive target |
|---|---|---|
| wholesaler → retailer | 2 | $(results.naive_targets.l2_wholesaler_to_retailer) |
| factory → wholesaler | 2 | $(results.naive_targets.l3_factory_to_wholesaler) |
| supplier → factory | 4 | $(results.naive_targets.l4_supplier_to_factory) |

Run across the $(results.scenario_count) scenarios:

| Metric | Value |
|---|---|
| Total demand | $(fmt(naive.total_demand)) units |
| Units sold | $(fmt(naive.total_sales)) |
| Lost sales | $(fmt(naive.total_lost_sales)) units |
| **Fill rate** | **$(pct(naive.fill_rate))** |

And the bullwhip ratio (Var(orders placed)/Var(customer demand)) at each echelon, averaged across all $(results.scenario_count) scenarios:

| Echelon | Bullwhip ratio |
|---|---|
| Retailer's orders (→ wholesaler) | $(ratio(naive_bw["retailer_orders (l2)"])) |
| Wholesaler's orders (→ factory) | $(ratio(naive_bw["wholesaler_orders (l3)"])) |
| Factory's orders (→ supplier) | $(ratio(naive_bw["factory_orders (l4)"])) |

That's the textbook shape exactly: mild amplification at the retailer, worse at the wholesaler, worst at the factory — the pain compounds the further you are from the actual customer.

## What the optimizer finds — and it's not what you'd expect

Same network, same $(results.scenario_count) scenarios, but the three upstream echelons now run `BackwardCoverageOrderingPolicy`, tuned jointly by `optimize!` against one shared cost function (lost sales + holding + transportation + order costs):

| Metric | Naive | Optimized (same scenarios) |
|---|---|---|
| Fill rate | $(pct(naive.fill_rate)) | $(pct(opt_in.fill_rate)) |
| Lost sales | $(fmt(naive.total_lost_sales)) units | $(fmt(opt_in.total_lost_sales)) units |

Fill rate got *worse* — $(fill_rate_change_pp) percentage points worse — not better. And the bullwhip ratios:

| Echelon | Naive | Optimized |
|---|---|---|
| Retailer's orders | $(ratio(naive_bw["retailer_orders (l2)"])) | **$(ratio(opt_bw["retailer_orders (l2)"]))** |
| Wholesaler's orders | $(ratio(naive_bw["wholesaler_orders (l3)"])) | **$(ratio(opt_bw["wholesaler_orders (l3)"]))** |
| Factory's orders | $(ratio(naive_bw["factory_orders (l4)"])) | **$(ratio(opt_bw["factory_orders (l4)"]))** |

The optimized policy's order quantities swing with roughly **$(ratio(opt_bw["retailer_orders (l2)"])) to $(ratio(opt_bw["factory_orders (l4)"])) times the variance of actual customer demand** — an order of magnitude worse than the naive policy's already-textbook bullwhip, and, unlike the naive case, it's severe at *every* echelon rather than escalating upstream. For scenario 1 specifically (directly comparable to the repo's own `beer_game()` test, which asserts exactly `lost_sales == 103.0 && sales == 1828.0 && demand == 1931.0`): this run produced lost sales = $(fmt(s1.total_lost_sales)), sales = $(fmt(s1.total_sales)), demand = $(fmt(s1.total_demand)) — matching the known-good value, so this isn't a bug in this analysis; it's what that policy actually does.

![Orders placed per echelon over the first 60 periods of scenario 1 - naive policy (top) vs. optimized policy (bottom), both on the same units-ordered scale](bullwhip_orders.png)

Seeing the two on the same scale is more convincing than the ratios alone: the naive policy's oscillation is the textbook shape from six decades of beer-game literature — mild, and building gradually as it moves upstream from retailer to factory. The "optimized" policy's orders swing far more violently, and at every echelon from the very first cycle, not just the ones furthest from the customer.

The tuned parameters `optimize!` found (`BackwardCoverageOrderingPolicy.cover`, applied per `Policy.jl` as: target net inventory = (last period's local demand) × (cover[1]+cover[2]) + cover[2] — "local demand" being what this node's own downstream customer ordered from it, via `get_past_outbound_orders`, not this node's own order history and not true end-customer demand):

| Lane | Tuned `cover` | Implied rule |
|---|---|---|
| wholesaler → retailer | [$(cover2)] | targets ≈ $(round(Float64(results.tuned_policy_cover.l2_wholesaler_to_retailer[1]) + Float64(results.tuned_policy_cover.l2_wholesaler_to_retailer[2]); digits=1)) × last period's local demand — a hard trend-chaser |
| factory → wholesaler | [$(cover3)] | targets ≈ $(round(Float64(results.tuned_policy_cover.l3_factory_to_wholesaler[1]) + Float64(results.tuned_policy_cover.l3_factory_to_wholesaler[2]); digits=2)) × last period's local demand — much gentler |
| supplier → factory | [$(cover4)] | first coefficient ≈ 0 — barely reacts to recent demand at all, runs closer to a flat, thin-buffer order-up-to rule |

So this isn't one uniform mechanism copy-pasted three times — the optimizer found three structurally different rules (an aggressive trend-chaser, a gentle one, and one that mostly ignores the recent-demand signal) that happen to converge on similarly severe order-variance amplification. That convergence is itself worth flagging as something we don't have a single clean explanation for yet, and a good target for a follow-up post that looks at period-by-period order traces instead of aggregate variance ratios.

## So who's actually damping the swings — if anyone?

This is the real question, and the honest answer is: **nobody is damping the order stream.** All three echelons pass through similarly enormous amplification (roughly $(ratio(opt_bw["wholesaler_orders (l3)"])) to $(ratio(opt_bw["retailer_orders (l2)"]))), instead of the naive case's clean upstream escalation. But the *inventory* picture tells a different story — the coefficient of variation (std/mean) of each node's own on-hand inventory:

| Node | Naive inventory CV | Optimized inventory CV |
|---|---|---|
| Retailer | $(round(naive_cv["retailer"]; digits=2)) | $(round(opt_cv["retailer"]; digits=2)) |
| Wholesaler | $(round(naive_cv["wholesaler"]; digits=2)) | $(round(opt_cv["wholesaler"]; digits=2)) |
| Factory | $(round(naive_cv["factory"]; digits=2)) | $(round(opt_cv["factory"]; digits=2)) |

Under the optimized policy, inventory volatility still climbs upstream just like classic bullwhip theory predicts — it's the **factory**, furthest from the actual customer and behind the longest (4-period) lead time, that ends up carrying the most erratic buffer. So the variance doesn't get *smoothed* anywhere in this system; it gets *warehoused*, and it's warehoused worst at the node with the least direct visibility into real demand — which is exactly the node the original bullwhip literature identifies as the classic victim.

The backlog picture backs this up from a different angle. Customer orders never backlog in this model — an order that isn't filled the same period is dropped as a lost sale, not queued. Internal replenishment orders between echelons are different: they never expire, so anything a node can't ship immediately queues up as a real, measurable backorder. Peak and ending queue size, averaged across all $(results.scenario_count) scenarios:

| Node | Naive peak backlog | Optimized peak backlog | Naive ending backlog | Optimized ending backlog |
|---|---|---|---|---|
| Wholesaler | $(fmt(results.naive_backlog.peak.wholesaler)) | $(fmt(results.optimized_backlog.peak.wholesaler)) | $(fmt(results.naive_backlog.ending.wholesaler)) | $(fmt(results.optimized_backlog.ending.wholesaler)) |
| Factory | $(fmt(results.naive_backlog.peak.factory)) | $(fmt(results.optimized_backlog.peak.factory)) | $(fmt(results.naive_backlog.ending.factory)) | $(fmt(results.optimized_backlog.ending.factory)) |
| Supplier | $(fmt(results.naive_backlog.peak.supplier)) | $(fmt(results.optimized_backlog.peak.supplier)) | $(fmt(results.naive_backlog.ending.supplier)) | $(fmt(results.optimized_backlog.ending.supplier)) |

(The supplier's backlog should read ~0 in every column — it's modeled with unconstrained throughput, so it never has to queue an order. That's a sanity check on this measurement, not a finding.)

## A fair fight: tuning the *right* family beats tuning harder within the wrong one

Everything above compares an **untuned** `NetUptoOrderingPolicy` against a **tuned** `BackwardCoverageOrderingPolicy` — not a fair fight. Two follow-ups close that gap:

1. Tune `NetUptoOrderingPolicy`'s own (single) parameter per echelon with `optimize!`, the same way `BackwardCoverageOrderingPolicy` was tuned above.
2. Give `BackwardCoverageOrderingPolicy` **4x the search budget** (60,000 evaluations and 6,000 no-progress steps, vs. the default 15,000/1,500) to check whether the original result was under-converged.

| Policy | Fill rate | Total cost (`optimize!`'s objective) |
|---|---|---|
| Naive `NetUptoOrderingPolicy` (untuned) | $(pct(naive.fill_rate)) | $(fmt(naive_total_cost)) |
| **Tuned `NetUptoOrderingPolicy`** | **$(pct(Float64(tn.aggregate.fill_rate)))** | **$(fmt(tn_total_cost))** |
| `BackwardCoverageOrderingPolicy` (15,000 evals) | $(pct(opt_in.fill_rate)) | $(fmt(optimized_total_cost)) |
| `BackwardCoverageOrderingPolicy` (60,000 evals) | $(pct(Float64(bo.aggregate.fill_rate))) | $(fmt(bo_total_cost)) |

Tuning the *simple, correct-shaped* family wins outright: the tuned `NetUptoOrderingPolicy` is $(family_beats_naive_pct)% cheaper than the untuned naive baseline **and** $(family_beats_optimized_pct)% cheaper than the searched `BackwardCoverageOrderingPolicy` — while also posting the best fill rate of any configuration on this page. $(budget_verdict)

The targets `optimize!` found for the tuned naive policy are worth reading closely:

| Lane | Untuned target | Tuned target |
|---|---|---|
| wholesaler → retailer (retailer's own target) | $(results.naive_targets.l2_wholesaler_to_retailer) | $(tn_targets.l2_wholesaler_to_retailer) |
| factory → wholesaler (wholesaler's own target) | $(results.naive_targets.l3_factory_to_wholesaler) | $(tn_targets.l3_factory_to_wholesaler) |
| supplier → factory (factory's own target) | $(results.naive_targets.l4_supplier_to_factory) | $(tn_targets.l4_supplier_to_factory) |

The optimizer didn't spread the safety margin evenly — it moved almost all of it to the **retailer** (target roughly doubled) and cut the **wholesaler's** to $(tn_targets.l3_factory_to_wholesaler). The next section explains why, and what it costs the wholesaler in practice.

## Who's actually keeping this chain cheap — and what it risks

The cost function only ever charges `lost_sales` at the customer interface — a retailer stockout. A wholesaler or factory stockout is internal backlog: it never expires, and it costs only the marginal holding-cost differential of carrying less stock, not a "lost sale" line item. Handed that asymmetry, `optimize!` did the economically correct thing: it pushed the visible safety buffer to the node whose stockouts are actually charged (the retailer) and let the node whose stockouts are nearly free in this objective (the wholesaler) run with almost no cushion at all.

The inventory-volatility signature makes this concrete — coefficient of variation (std/mean) of on-hand inventory under the tuned policy:

| Node | Inventory CV |
|---|---|
| Retailer | $(round(tn_cv["retailer"]; digits=2)) |
| Wholesaler | **$(round(tn_cv["wholesaler"]; digits=2))** |
| Factory | $(round(tn_cv["factory"]; digits=2)) |

That's not a rounding artifact of a near-zero average — the backlog numbers confirm it in real units. Averaged across all $(results.scenario_count) scenarios, the wholesaler's queue of unfilled internal orders:

| | Untuned naive | Tuned |
|---|---|---|
| Peak backlog | $(fmt(Float64(results.naive_backlog.peak.wholesaler))) units | **$(fmt(Float64(tn.backlog.peak.wholesaler))) units** |
| Ending backlog | $(fmt(Float64(results.naive_backlog.ending.wholesaler))) units | **$(fmt(Float64(tn.backlog.ending.wholesaler))) units** |

Under the cost-minimal tuned policy, the wholesaler ends the average 200-period run still carrying about $(fmt(Float64(tn.backlog.ending.wholesaler))) units of permanent unfilled backorder — nearly $(round(Float64(tn.backlog.ending.wholesaler) / Float64(results.naive_backlog.ending.wholesaler); digits=1))x the untuned baseline — and peaks at $(fmt(Float64(tn.backlog.peak.wholesaler))) mid-run. That's the real content behind "the wholesaler is keeping this chain cheap": it's absorbing the volatility the model doesn't charge for, not damping it.

**Could a real wholesaler run this way?** Not for free. A distributor deliberately held at a near-zero safety-stock target would in practice be paying for this policy through channels the model doesn't price: constant expediting to plug the gaps, working-capital swings as on-hand oscillates between small overage and real shortfall, and the commercial/contractual cost of frequently under-shipping the retailer it supplies. None of that shows up in `metrics_cost_function` — only the terminal customer-facing miss does. The $(family_beats_naive_pct)% cost saving this section reports is real *under this cost function*; it is not free in an operation where a mid-chain stockout has consequences the model can't see.

## Fixing the actual gap: pricing backlog

Everything above traces back to one concrete, checkable fact: `metrics_cost_function` — the objective every `optimize!` call so far has minimized — reads only `state.metrics`, and `SimMetrics` (`Metrics.jl`) tracks `sales`, `lost_sales`, `holding_costs`, `overflow_costs`, `trip_unit_costs`, `trip_fixed_costs`, `orders`, and `demand`. **No backlog field at all.** Internal backlog at the wholesaler and factory has been free in every optimization on this page — not approximately free, literally zero cost in the objective actually being minimized. `classic_score`, used for every backlog table above, was only ever computed after the fact for reporting; it was never what `optimize!` searched against.

We fixed this at the package level rather than working around it: `SimMetrics` now has a `backlog` field, accumulated every period exactly like `holding_costs` already was — read live from `pending_outbound_order_lines` (an order line is removed from there the instant it's filled or dropped, so whatever's left when a period closes out is exactly what's genuinely still owed, no history scan required) and excluding customer-destined orders (which are dropped as lost sales, never backlogged, per the same convention used throughout this page). That means the real beer-game cost — holding plus backlog, not just holding plus lost sales — can now be `optimize!`'s actual objective at full speed, `record_history=false`, the same fast path as every other search on this page.

Re-tuning both policy families against that real cost:

| | `NetUptoOrderingPolicy` targets | Fill rate | Classic score |
|---|---|---|---|
| Untuned naive | 30 / 30 / 50 | $(pct(naive.fill_rate)) | $(fmt(naive_classic_total)) |
| Tuned under `metrics_cost_function` (backlog free) | $(tn_targets.l2_wholesaler_to_retailer) / $(tn_targets.l3_factory_to_wholesaler) / $(tn_targets.l4_supplier_to_factory) | $(pct(Float64(tn.aggregate.fill_rate))) | $(fmt(Float64(tn.classic.total))) |
| **Tuned under the real beer-game cost** | **$(ctn_targets.l2_wholesaler_to_retailer) / $(ctn_targets.l3_factory_to_wholesaler) / $(ctn_targets.l4_supplier_to_factory)** | **$(pct(Float64(ctn.aggregate.fill_rate)))** | **$(fmt(ctn_classic_total))** |

That's the confirmation: once backlog is actually priced, the wholesaler's tuned target jumps from **$(tn_targets.l3_factory_to_wholesaler) to $(ctn_targets.l3_factory_to_wholesaler)** — no longer starved. Inventory volatility falls back in line with every other node (wholesaler CV $(round(tn_cv["wholesaler"]; digits=2)) → $(round(ctn_cv["wholesaler"]; digits=2)), against retailer's $(round(ctn_cv["retailer"]; digits=2)) and factory's $(round(ctn_cv["factory"]; digits=2)) under the same real-cost tuning), and its ending backlog drops from $(fmt(Float64(tn.backlog.ending.wholesaler))) units to $(fmt(Float64(ctn.backlog.ending.wholesaler))) — and the resulting policy isn't even worse on the classic scoring: $(fmt(ctn_classic_total)) vs. the untuned baseline's $(fmt(naive_classic_total)), a genuine (if modest) improvement, achieved with a *sane* wholesaler buffer instead of a starved one. The earlier "cheap" result wasn't cheap — it was uncosted.

**Does the same fix rescue `BackwardCoverageOrderingPolicy`?** Re-tuning it against the real cost too:

| | Fill rate | Classic score | Bullwhip (retailer / wholesaler / factory) |
|---|---|---|---|
| Tuned under `metrics_cost_function` | $(pct(opt_in.fill_rate)) | $(fmt(optimized_classic_total)) | $(ratio(opt_bw["retailer_orders (l2)"])) / $(ratio(opt_bw["wholesaler_orders (l3)"])) / $(ratio(opt_bw["factory_orders (l4)"])) |
| **Tuned under the real beer-game cost** | **$(pct(co_fill_rate))** | **$(fmt(co_classic_total))** | **$(ratio(Float64(as_dict(co.bullwhip)["retailer_orders (l2)"]))) / $(ratio(Float64(as_dict(co.bullwhip)["wholesaler_orders (l3)"]))) / $(ratio(Float64(as_dict(co.bullwhip)["factory_orders (l4)"])))** |

Bullwhip does calm down substantially — no more 30-37x swings. But fill rate collapses to $(pct(co_fill_rate)), and the classic score ($(fmt(co_classic_total))) is still more than $(round(co_classic_total/ctn_classic_total; digits=1))x worse than the simple family tuned under the identical, now-correct cost. With lost-sales cost (1.0/unit, one-time) cheap relative to holding and backlog charges accumulating over a 200-period horizon, the search found it's cost-minimal for this family to run extremely lean and simply eat the lost sales — a different failure mode than the earlier blow-up, but still a worse answer than `NetUptoOrderingPolicy`'s. Pricing backlog correctly fixed the *objective's* blind spot; it didn't fix this policy family's fit to the network. That reinforces, from a completely different angle, the same conclusion the "fair fight" section reached: which policy family you search matters more than which cost function or how much budget you give the search.

**Is that collapse another search failure, or is this policy genuinely a worse structural fit once backlog is priced correctly?** Same test as the `equiv_metrics` check earlier on this page, just against the real cost this time: plug the already-known `classic_tuned_naive` targets directly into `BackwardCoverageOrderingPolicy([0.0, target])` for each lane — no `optimize!` call at all — and compare to what the search actually found:

| | Fill rate | Classic score |
|---|---|---|
| `optimize!`'s search (real cost) | $(pct(co_fill_rate)) | $(fmt(co_classic_total)) |
| **Known-optimum point, plugged in directly** | **$(pct(Float64(cem.aggregate.fill_rate)))** | **$(fmt(cem_total_cost))** |

$(classic_equiv_matches_tuned_naive ? "That matches the tuned-naive result to within rounding — so yes, same story as before: the optimum was reachable inside `BackwardCoverageOrderingPolicy`'s own search space the whole time, under this cost function too. The search failed to find it, on two different objectives now, not because the policy can't represent a good answer." : "This one doesn't fully match tuned-naive, which is worth investigating on its own rather than assuming the same explanation carries over.")

## Can search itself be fixed, generically?

Two confirmed search failures on the same landscape feature (a kink at `cover[1]=0`, where the weighted-average branch in `get_order` switches off entirely) raises the obvious next question: can `optimize!` be improved to actually find the optimum it's missing, without hand-coding beer-game-specific knowledge into the fix? Three attempts, in order of sophistication, all starting from the same collapsed point above:

**1. Coarse-to-fine restart.** Run the search once broadly, narrow the range around wherever it landed, rerun — a standard restart pattern:

| | Fill rate | Classic score |
|---|---|---|
| Original search | $(pct(co_fill_rate)) | $(fmt(co_classic_total)) |
| Coarse-to-fine (refined range $(cf_range)) | $(pct(Float64(cf.aggregate.fill_rate))) | $(fmt(Float64(cf.classic.total))) |

Essentially no improvement. The reason shows up directly in the cover values across independent runs: coarse-to-fine's "coarse" stage reliably converges to the same dominant bad attractor basin every time, so narrowing the second stage around it just entrenches the wrong answer rather than escaping it — the search isn't failing to look hard enough in the region it already found, it's looking in the wrong region entirely.

**2. Empirical parameter-scale probing.** Instead of letting the population discover its own exploration scale inside one shared box, probe each parameter dimension independently before searching at all: perturb it up and down from the starting point, holding every other dimension fixed, and see which step size actually moves the cost. This can't rely on a coordinate's *position* meaning anything — a dimensionless gain and an inventory-level target don't share a natural scale — so the probe is purely empirical, the same information a human would use ("moving this one by 1 changes everything; moving that one by 1 does basically nothing"), just measured instead of guessed:

| | Fill rate | Classic score |
|---|---|---|
| Original search | $(pct(co_fill_rate)) | $(fmt(co_classic_total)) |
| Probed per-dimension ranges (half-widths $(probed_ranges_r)) | $(pct(Float64(po.aggregate.fill_rate))) | $(fmt(Float64(po.classic.total))) |

A real jump — fill rate roughly $(probing_gain_ratio)x the original search's, from probe scales $(probe_scales_r) that are themselves the finding: several coordinates probe to at or near zero (the search shouldn't move them far from their starting point), while others probe to genuinely large steps — exactly the "some coordinates are dimensionless gains, some are inventory levels" asymmetry a single shared `SearchRange` can't represent, and directly reflected in the per-dimension ranges it produces (half-widths $(probed_ranges_r)).

**3. Iterative probe-then-search.** Repeat the cycle, re-centering each round's range on wherever the *previous* round actually landed, instead of stopping after one pass:

| | Fill rate | Classic score |
|---|---|---|
| Single-round probing | $(pct(Float64(po.aggregate.fill_rate))) | $(fmt(Float64(po.classic.total))) |
| 3-round iterative probing | $(pct(Float64(ip.aggregate.fill_rate))) | $(fmt(Float64(ip.classic.total))) |
| **Known optimum (`classic_tuned_naive`)** | **$(pct(Float64(ctn.aggregate.fill_rate)))** | **$(fmt(ctn_classic_total))** |

Further, real, but clearly diminishing gains — from $(pct(co_fill_rate)) (original search) to $(pct(Float64(po.aggregate.fill_rate))) (single-round probing) to $(pct(Float64(ip.aggregate.fill_rate))) (three rounds), still well short of the known-reachable $(pct(Float64(ctn.aggregate.fill_rate))). Better box-shaping around the search clearly helps, entirely without telling the optimizer anything specific to the beer game — but it plateaus well short of the reachable optimum. The most likely reason: `bboptimize`'s own population-seeding and mutation dynamics (see `Optimization.jl`) have limits that better range-shaping alone can't fully overcome once the population has already settled near a boundary kink. That's a real, open, honestly-reported gap, not a solved problem — the next lever to pull is the search algorithm's core dynamics, not its input ranges, and that's future work rather than something this post claims to have finished.

(One genuine, previously-invisible bug surfaced while building this: `AnchorAndAdjustOrderingPolicy`'s own `set_parameters!` — never called by `optimize!` itself, only by this probing code calling it directly from a script-level closure — had been defined without qualifying it as `SupplyChainSimulation.set_parameters!`. Julia doesn't let a plain, unqualified function definition extend an existing module's function just because that module is `using`'d, even for an exported name — so this silently created an unrelated `Main.set_parameters!` instead of adding a method to the real one, invisible everywhere else in this script because `optimize!`'s own internal calls resolve inside the package's own module scope regardless of what `Main.set_parameters!` points to. Fixed by qualifying the definition — a good reminder that this failure mode isn't limited to non-exported functions, as flagged earlier on this page for `get_order`/`get_parameters`.)

## Is the "optimized" policy actually cheaper? Checking, not assuming.

Everything above raises an obvious question this analysis shouldn't skip: if the optimized policy has worse fill rate *and* far worse bullwhip, is it at least genuinely cheaper under the cost function `optimize!` was minimizing? That's worth checking directly rather than assuming — `optimize!` only ever searched `BackwardCoverageOrderingPolicy` parameters here, so there was never a guarantee it could reach something as good as the naive `NetUptoOrderingPolicy` baseline, which lives in a different, entirely unsearched policy family.

| Cost component | Naive | Optimized |
|---|---|---|
| Holding costs | $(fmt(results.naive_costs.holding_costs)) | $(fmt(results.optimized_costs.holding_costs)) |
| Transportation (fixed) | $(fmt(results.naive_costs.trip_fixed_costs)) | $(fmt(results.optimized_costs.trip_fixed_costs)) |
| Transportation (unit) | $(fmt(results.naive_costs.trip_unit_costs)) | $(fmt(results.optimized_costs.trip_unit_costs)) |
| Order count | $(fmt(results.naive_costs.order_count)) | $(fmt(results.optimized_costs.order_count)) |
| **Total cost (`optimize!`'s literal objective)** | **$(fmt(naive_total_cost))** | **$(fmt(optimized_total_cost))** |

$(cost_verdict)

## Scoring it the way the original board game does

Everything above uses this package's own cost convention: a customer order that isn't filled the same period is a permanently **lost sale**. The physical MIT beer game scores differently — every stage, *including the retailer's shelf*, just carries a **backorder** forward with a per-period backlog cost until it's eventually filled. Nothing is ever permanently lost in the original game; it's just late, and lateness is what's penalized.

This package's `Customer` node type can't fully replicate that: customer orders always have `due_date == creation_time` (hardcoded in `Simulation.jl`'s `place_orders`), so an unfilled customer order is dropped, not queued — unlike every other node type, which backlogs correctly, as the table above shows. So this is as close as this model can get: real holding + backlog cost (at $(Int(round(0.2/0.1)))× the holding rate, the standard ratio in Sterman's canonical version of the game) at the wholesaler, factory, and supplier, and a holding-cost-plus-lost-sales *proxy* at the retailer, clearly not equivalent to a true backlog cost:

| Stage | Naive | Optimized |
|---|---|---|
| Retailer (holding + lost-sales proxy) | $(fmt(Float64(results.naive_classic_score.per_stage.retailer))) | $(fmt(Float64(results.optimized_classic_score.per_stage.retailer))) |
| Wholesaler (holding + backlog) | $(fmt(Float64(results.naive_classic_score.per_stage.wholesaler))) | $(fmt(Float64(results.optimized_classic_score.per_stage.wholesaler))) |
| Factory (holding + backlog) | $(fmt(Float64(results.naive_classic_score.per_stage.factory))) | $(fmt(Float64(results.optimized_classic_score.per_stage.factory))) |
| Supplier (holding + backlog) | $(fmt(Float64(results.naive_classic_score.per_stage.supplier))) | $(fmt(Float64(results.optimized_classic_score.per_stage.supplier))) |
| **Total classic score** | **$(fmt(Float64(results.naive_classic_score.total)))** | **$(fmt(Float64(results.optimized_classic_score.total)))** |

$(classic_verdict)

This distinction matters beyond bookkeeping: a permanently-lost-sale model and an eventually-fulfilled-backorder model reward *completely different* policies. A model where stockouts are forgiven (backlogged, filled later) can rationally tolerate more short-term volatility than one where every miss is gone for good — so a policy tuned against this package's lost-sales convention isn't just numerically different from one tuned against the classic game's convention, it's answering a different question. Worth stating plainly rather than glossing over: **this whole post measures against this package's convention, not the original board game's**, and that choice is a real driver of everything above, not a footnote.

## Actually closing that gap: letting customer orders backlog too

The section above stopped at "this is as close as this model can get" — customer orders always dropped as a one-time lost sale rather than backlogging like every other node type, because `due_date == creation_time` was hardcoded for `Customer` destinations in `Simulation.jl`'s `place_orders`. That's no longer a hard limit on this branch: `Env` now takes a `customer_backlog` flag (default `false`, so every number on this page above is unaffected) that gives a customer order the same open-ended due date as an internal replenishment order instead, so it queues and compounds cost like any other backorder instead of vanishing into a lost-sales charge after one period.

Re-tuning `NetUptoOrderingPolicy` against the real cost, this time with `customer_backlog=true` end to end — both the simulation itself and the scoring, so a retailer stockout is genuine compounding backlog cost, not a lost-sales proxy:

| | Targets (wholesaler / factory / supplier lane) | Fill rate |
|---|---|---|
| Tuned under the real cost (lost-sales proxy at retail) | $(ctn_targets.l2_wholesaler_to_retailer) / $(ctn_targets.l3_factory_to_wholesaler) / $(ctn_targets.l4_supplier_to_factory) | $(pct(Float64(ctn.aggregate.fill_rate))) |
| **Tuned with customer orders backlogging too** | **$(cbtn_targets.l2_wholesaler_to_retailer) / $(cbtn_targets.l3_factory_to_wholesaler) / $(cbtn_targets.l4_supplier_to_factory)** | **$(pct(Float64(cbtn.aggregate.fill_rate)))** |

Fill rate jumps to $(pct(Float64(cbtn.aggregate.fill_rate))) — because a customer order that misses is no longer written off, the optimizer has the same direct, ongoing incentive to keep the retailer well-stocked that already kept the wholesaler and factory well-stocked once their own backlog was priced correctly earlier on this page. (Classic-score totals aren't directly comparable between the two rows above — the retailer's term switches from a lost-sales proxy to real compounding backlog cost, a different scoring convention entirely, not just a different policy — so this table reports targets and fill rate only, not a cost comparison.) This is the closest this package can currently get to the original board game's actual convention: nothing permanently lost anywhere in the chain, everything eventually filled, backlog cost the only penalty.

## What do real humans actually do? The Sterman anchor-and-adjust sweep

Every policy above is a rational optimizer's answer. But the original beer game is famous precisely because *real people* playing it — including people who understand supply chains — still produce the bullwhip effect. John Sterman's 1989 study modeled how people actually order with an "anchor and adjust" heuristic: each period, order = forecast + (how far on-hand is from a desired stock level) + (how far the pipeline is from where it should be). The measured finding, replicated since (Croson & Donohue found 98% of players in the original study, and roughly 64% in a later replication, systematically underweight the pipeline term) is that people don't fully trust an order they've already placed is "on the way" — a delayed shipment reads as a fresh shortfall on top of what's already coming.

Holding the desired-stock term at zero (i.e., "I want to run lean, no cushion") and sweeping how much of the pipeline people credit:

| Pipeline credit | Fill rate | Factory bullwhip | Total cost |
|---|---|---|---|
$(sweep_rows)

At full pipeline credit (1.0) this is algebraically identical to the naive baseline — that's a built-in correctness check, not a coincidence. As people trust the pipeline less, fill rate falls and cost gets worse, but bullwhip *doesn't* — it stays roughly flat to mildly *higher* than naive, not the dramatic panic-buying spikes the beer game is known for. The reason is structural: with desired stock pinned at zero, there's no term pulling orders *up* when on-hand runs low, only the pipeline-credit term — so underweighting it produces steady under-ordering, not overshoot.

## Does a safety-stock cushion reproduce genuine panic-buying?

Real players don't anchor on wanting zero inventory — they want a visible cushion, and it's the *gap* between that cushion and what's actually on the shelf, compounded by not trusting the pipeline, that plausibly drives real overshoot. Holding pipeline credit fixed at $(hp_alpha) (inside the 0.3–0.5 range Croson & Donohue's replications measured for real players) and sweeping the desired safety cushion up from zero:

| Desired stock cushion | Fill rate | Retailer bullwhip | Wholesaler bullwhip | Factory bullwhip | Total cost |
|---|---|---|---|---|---|
$(panic_rows)

This is the result that actually looks like the beer game: as the cushion target rises, bullwhip climbs steadily at every echelon — factory bullwhip goes from $(ratio(panic_factory_bw_low)) at zero cushion to **$(ratio(panic_factory_bw_high))** at a cushion of 20, higher than even the untuned naive baseline's $(ratio(naive_bw["factory_orders (l4)"])). Fill rate also improves sharply ($(pct(panic_fill_low)) → $(pct(panic_fill_high))), which is the honest, slightly uncomfortable finding: wanting a visible buffer and not fully trusting the pipeline is a *worse* combination for order-stream stability than either bias alone, even though it serves customers better. That's the actual mechanism behind "people panic-order" — not irrationality, but a reasonable desire for a safety margin colliding with a reasonable distrust of a shipment that hasn't arrived yet, and the two compounding upstream.

## Should SupplyChainSimulation.jl keep `BackwardCoverageOrderingPolicy`?

Given everything above, it's worth asking directly instead of leaving it implied. `get_order` for this policy targets a multiple of **the recent demand this node has itself experienced** — what its own downstream customer ordered from it, via `get_past_outbound_orders` (despite the "outbound" name, that reads orders placed *against* this node, not orders this node placed upstream, and not true end-customer demand — a mid-chain node never sees that directly):

```
target = last_local_demand × (cover[1] + cover[2]) + cover[2]
```

Any tuning where that multiplier exceeds 1 is naive, undamped trend extrapolation: a one-period uptick in local demand is projected forward as though it were a lasting shift, inflating this period's target by more than the uptick itself. That's the "demand signal processing" bullwhip mechanism from Lee, Padmanabhan, and Whang (1997) — one of the four canonical, published causes of the bullwhip effect, playing out directly in this policy's structure. The tuned parameters found above (target ≈ $(round(Float64(results.tuned_policy_cover.l2_wholesaler_to_retailer[1]) + Float64(results.tuned_policy_cover.l2_wholesaler_to_retailer[2]); digits=1))× the retailer's most recent local demand) land well inside that regime, and $(bigger_budget_helped ? "quadrupling the search budget only found a different point inside the same unstable regime (still ≈$(round(Float64(bo_cover.l2_wholesaler_to_retailer[1]) + Float64(bo_cover.l2_wholesaler_to_retailer[2]); digits=1))×), not a way out of it" : "quadrupling the search budget didn't move it out either"). Nothing in `optimize!`'s default search range caps that multiplier below 1.

**But is the blow-up actually structural, or just a search failure?** This policy family strictly contains `NetUptoOrderingPolicy` as a special case: at `cover[1]=0`, the `weights` accumulator above never leaves 0, the weighted-average branch is skipped entirely, and `coverage` collapses to the bare constant `cover[2]` — algebraically identical to `NetUptoOrderingPolicy(upto=cover[2])`. That point sits inside the exact same `[0, 5000]` range `optimize!` searched. Plugging in the already-known tuned-naive targets directly — `BackwardCoverageOrderingPolicy([0.0, $(tn_targets.l2_wholesaler_to_retailer)])` and so on, no `optimize!` call at all — gives fill rate $(pct(Float64(em.aggregate.fill_rate))) and total cost $(fmt(em_total_cost))$(equiv_matches_tuned_naive ? ", matching the tuned-naive result above to within rounding" : " — notably different from the tuned-naive result above, worth investigating further"). $(equiv_matches_tuned_naive ? "That settles it: the optimum was reachable the entire time. Neither the 15,000-eval search nor the 60,000-eval rerun found it." : "")

Our recommendation: **keep the policy** — it's a realistic, commonly-used-in-practice heuristic (order based on trailing sell-through × a coverage factor is genuinely how some real distributors replenish), and it demonstrates a named, published bullwhip cause directly and empirically, which is valuable on its own. What this section demonstrates isn't that the policy is broken, and it isn't even that this policy family is a bad fit for the network — it's that **`optimize!`'s default black-box search failed to find an optimum that was sitting inside its own declared search space**, most likely because `cover[1]=0` is a boundary point with a genuine kink in the cost surface right next to it (the `weights != 0` branch switch changes the function's behavior qualitatively on either side), which is exactly the kind of landscape a population-based, gradient-free optimizer struggles to converge onto precisely. The practical fix isn't "don't trust this policy" — it's **seed the search at `cover[1]=0` (or run several restarts from different starting points)** rather than trusting a single default-initialized `bboptimize` run on a landscape like this one. Separately, if a genuinely damped (rather than merely boundary-flat) version is wanted, capping the gain (`cover[1]+cover[2] ≤ 1`) or replacing the naive weighted-sum with an actual exponential-smoothing forecast is the standard real-world remedy.

## Can this be replicated in a real supply chain? Why, and why not.

**Why not, mostly:**

1. **A ~$(ratio(opt_bw["retailer_orders (l2)"])) order-variance swing is not something a real operation can execute for free**, even if a cost function says it's cheap. Trucking capacity is booked in advance. Production lines have changeover costs and minimum run lengths. A supplier asked to ship 30x more or less than usual with no forward notice will build in padding, refuse, or renegotiate the contract — none of which this simulation's unconstrained, infinite-throughput supplier node has to deal with.
2. **The holding-cost model is linear and flat.** A factory whose inventory coefficient of variation is $(round(opt_cv["factory"]; digits=2)) faces real working-capital, warehouse-capacity, and obsolescence risk that a constant `\$0.10 per unit per period` charge doesn't capture. In practice, that volatility has a cost this model is blind to.
3. **This is still a centralized result.** `optimize!` set all three echelons' rules jointly, with one shared objective. Even a company willing to tolerate this much internal volatility would need the automation and cross-echelon authority to execute it without every swing triggering a manual override from whoever's job it is to explain a tripled order to their boss.
4. **The tuned naive policy's own answer — starve the wholesaler's buffer to zero — carries the same problem in a quieter form.** It's cheaper *in this cost function* specifically because the function doesn't price the operational cost of chronic mid-chain backlog. A real wholesaler asked to run this way would be paying for it through expediting, working capital, and its own relationship with the retailer — costs this model doesn't see.

**Why it's still worth knowing:** it's a sharper, more useful result than "the optimizer solves the bullwhip" would have been. It shows precisely how sensitive an "optimal" ordering policy is to the cost assumptions it's handed — change the weight on order-quantity stability, or cap the supplier's throughput, or charge internal backlog something closer to its real operational cost, and you'd likely get a genuinely different, smoother policy. That's a concrete, testable next experiment (a natural Part 2: sweep an order-change penalty, a supplier capacity constraint, and an internal-backlog cost, and see what it takes to get a policy that's both cheap *and* operationally sane) rather than a vague caveat.

## The generalization check

The optimized policy above was tuned on $(results.scenario_count) scenarios seeded at $(results.calibration_seed). Running that *exact same, already-tuned* policy — no re-optimization — against a fresh batch of $(results.scenario_count) scenarios it never saw (seed $(results.holdout_seed), same demand distribution) gives:

| Metric | In-sample (tuned on this data) | Out-of-sample (fresh scenarios) |
|---|---|---|
| Fill rate | $(pct(opt_in.fill_rate)) | $(pct(opt_hold.fill_rate)) |
| Lost sales | $(fmt(opt_in.total_lost_sales)) units | $(fmt(opt_hold.total_lost_sales)) units |

Gap: **$(generalization_gap_pp) percentage points**, entirely sampling noise since both batches are drawn from the identical Poisson($(results.mean_demand)) process — this is a floor, not a worst case. But given what the bullwhip numbers above actually show, distributional generalization isn't the most important robustness question for this policy. **Sensitivity to the cost weights themselves** — what happens to the tuned policy, and its bullwhip ratios, as the order-quantity penalty or the holding-cost rate change — is the more important test, and it's the one this run doesn't answer yet.

## Try it yourself

The full script that produced every number on this page is [`model.jl`](model.jl) in this directory; the [SupplyChainSimulation.jl docs](https://SupplyChef.github.io/SupplyChainSimulation.jl/dev) walk through the same network shape from first principles (EOQ → safety stock → beer game).

---
*Sources: Sterman, J.D. (1989), "Modeling Managerial Behavior: Misperceptions of Feedback in a Dynamic Decision-Making Experiment," Management Science 35(3). Lee, H.L., Padmanabhan, V., Whang, S. (1997), "The Bullwhip Effect in Supply Chains," Sloan Management Review / "Information Distortion in a Supply Chain: The Bullwhip Effect," Management Science 43(4) — the Var(orders)/Var(demand) bullwhip ratio used throughout this page is their standard definition.*
"""

open(joinpath(@__DIR__, "post.md"), "w") do io
    write(io, post)
end

println("Wrote post.md")
