#=
Feasibility spike: can ForwardDiff differentiate optimize!'s actual cost
function w.r.t. BackwardCoverageOrderingPolicy's parameters, by
differentiating the real simulation directly (pathwise/Infinitesimal
Perturbation Analysis - Glasserman & Tayur 1995 established this is valid
almost everywhere for base-stock-style policies in multi-echelon systems),
rather than a surrogate or a hand-derived analytical approximation?

Motivation: :nelder_mead currently wins benchmark/compare_optimizers.jl's
comparison on both problems, but its simplex-based search is provably less
sample-efficient than a gradient-based local method once real gradients are
available. If direct AD works, a follow-up could seed L-BFGS (via SciML's
Optimization.jl + OptimizationOptimJL) from :nelder_mead's/:cma_es's result
for a final gradient-guided polish within the same evaluation budget.

This is deliberately NOT wired into optimize! or any test - it is a
standalone, throwaway check of whether AD gets through the real
minimize!/simulate() pipeline at all, and if not, exactly where it fails
(the error message IS the result, not a solved problem). Two things were
already changed in src/ specifically to give this its best chance:
- minimize!'s `x` parameter loosened from AbstractVector{Float64} to
  AbstractVector{<:Real} (Optimization.jl)
- BackwardCoverageOrderingPolicy's `cover` field loosened from
  Array{Float64,1} to Vector{Real}, set_parameters! loosened to match, and
  the order-quantity computation's ceil(Int, ...) truncation replaced
  with a type-dispatched _order_quantity that only applies the integer
  truncation for AbstractFloat (i.e. every existing, non-AD caller - zero
  behavior change there) and passes other Real subtypes (in practice,
  ForwardDiff.Dual) through continuously (Policy.jl)

Anything beyond those two files (State.jl, Simulation.jl, Model.jl) is
untouched - if AD fails there, this script's error output is exactly the
signal needed to find the next blocker, the same way every other bug this
session was found: from a real error, not from reading the whole engine
speculatively upfront.

Run with:
    julia --project=benchmark benchmark/ad_spike.jl
=#

using ForwardDiff
using Random
using Distributions: Poisson
using SupplyChainModeling
using SupplyChainSimulation

function build_problem()
    horizon = 20

    product = Product("product")
    supplier = Supplier("supplier")
    storage = Storage("storage")
    add_product!(storage, product; unit_holding_cost=1.0)
    customer = Customer("customer")

    l1 = Lane(storage, customer)
    l2 = Lane(supplier, storage)

    network = SupplyChain(horizon)
    add_supplier!(network, supplier)
    add_storage!(network, storage)
    add_customer!(network, customer)
    add_product!(network, product)
    add_lane!(network, l1)
    add_lane!(network, l2)
    add_demand!(network, customer, product, rand(Poisson(10), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)

    policy = BackwardCoverageOrderingPolicy([10.0, 20.0])
    policies = Dict((l2, product) => policy)

    return network, policies, policy
end

function main()
    Random.seed!(1)
    network, policies, policy = build_problem()

    sorted_keys = sort(collect(keys(policies)); by = k -> (string(k[1]), k[2].name))
    unique_policies = unique([policies[k] for k in sorted_keys])
    x0 = convert(Array{Float64, 1}, vcat([SupplyChainSimulation.get_parameters(p) for p in unique_policies]...))

    initial_states = SupplyChainSimulation.State.([network])
    envs = [SupplyChainSimulation.Env(network, initial_states, policies; record_history=false)]

    f = x -> SupplyChainSimulation.minimize!(policies, unique_policies, envs, initial_states, x; cost_function=metrics_cost_function)

    println("x0 = $x0")
    println("f(x0) = $(f(x0))")

    println("\nAttempting ForwardDiff.gradient(f, x0)...")
    grad = ForwardDiff.gradient(f, x0)
    println("SUCCESS: gradient = $grad")
end

main()
