#=
CPU (wall-clock) sampling profile of the actual beer_game() test function
(test/policy-beergame-tests.jl) - unlike profile_beergame.jl, which drives
minimize! directly with a handful of fixed candidate vectors as a cheap
proxy, this profiles the real BlackBoxOptim search (optimize!'s default
MaxFuncEvals=15000) plus the final simulate() calls after optimization, so
sample-count shares here reflect beer_game()'s actual wall-clock
composition instead of an isolated stand-in for it.

Run with:
    julia --project=benchmark benchmark/profile_beergame_direct.jl

Also used by .github/workflows/profile.yml to report results without
anyone needing to run this by hand.
=#

using Random
using Distributions: Poisson
using SupplyChainModeling
using SupplyChainSimulation
using Profile

# Mirrors test/policy-beergame-tests.jl's beer_game() exactly, plus a
# `params` passthrough (not present on the real test function) so the
# warmup call below can compile every code path optimize! exercises
# without paying for the real 15000-eval search twice.
function beer_game(; scenario_count=30, optimize=true, params=Dict{Symbol, Float64}())
    Random.seed!(3)

    product = SupplyChainModeling.Product("product")

    customer = Customer("customer")
    retailer = Storage("retailer")
    add_product!(retailer, product; unit_holding_cost=0.1, initial_inventory=20)
    wholesaler = Storage("wholesaler")
    add_product!(wholesaler, product; unit_holding_cost=0.1, initial_inventory=20)
    factory = Storage("factory")
    add_product!(factory, product; unit_holding_cost=0.1, initial_inventory=20)
    supplier = Supplier("supplier")

    horizon = 200

    l = Lane(retailer, customer; unit_cost=0)
    l2 = Lane(wholesaler, retailer; unit_cost=0, time=2)
    l3 = Lane(factory, wholesaler; unit_cost=0, time=2)
    l4 = Lane(supplier, factory; unit_cost=0, time=4)

    policy2 = BackwardCoverageOrderingPolicy([0.0, 0.0])
    policy3 = BackwardCoverageOrderingPolicy([0.0, 0.0])
    policy4 = BackwardCoverageOrderingPolicy([0.0, 0.0])

    n() = begin
        network = SupplyChain(horizon)
        add_supplier!(network, supplier)
        add_storage!(network, retailer)
        add_storage!(network, wholesaler)
        add_storage!(network, factory)
        add_customer!(network, customer)
        add_product!(network, product)
        add_lane!(network, l)
        add_lane!(network, l2)
        add_lane!(network, l3)
        add_lane!(network, l4)
        add_demand!(network, customer, product, rand(Poisson(10), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)
        return network
    end

    initial_states = [n() for i in 1:scenario_count]

    policies = Dict((l2, product) => policy2,
                     (l3, product) => policy3,
                     (l4, product) => policy4)

    if optimize
        SupplyChainSimulation.optimize!(policies, initial_states...; params=params, cost_function=metrics_cost_function, record_history=false)
    end

    final_states = [simulate(initial_state, policies) for initial_state in initial_states]
    return final_states
end

# Cheap warmup (JIT-compile everything) outside the profiled region: small
# scenario_count and a tiny MaxFuncEvals keep this fast without changing
# which code paths get compiled - the real profiled call below still uses
# optimize!'s actual default (15000).
println("Warming up...")
@time beer_game(scenario_count=2, params=Dict(:MaxFuncEvals => 50.0))

println("Warmup done. Profiling one full beer_game() call (scenario_count=30, the same shape as the actual test)...")

Profile.clear()
@time @profile beer_game()

flat_io = IOBuffer()
Profile.print(flat_io; format=:flat, sortedby=:count, C=false, mincount=5)
flat_lines = split(String(take!(flat_io)), '\n')

tree_io = IOBuffer()
Profile.print(tree_io; format=:tree, sortedby=:count, C=false, mincount=15, maxdepth=25)
tree_lines = split(String(take!(tree_io)), '\n')

# Profile.print(sortedby=:count) sorts rows ascending (lowest count first),
# so the interesting (highest-count) rows are at the *end* of the output,
# not the start - take the header plus the last 80 data rows, and reverse
# the data rows so the highest count reads first.
header_idx = findfirst(l -> occursin("=====", l), flat_lines)
header = isnothing(header_idx) ? String[] : flat_lines[1:header_idx]
data = filter(!isempty, isnothing(header_idx) ? flat_lines : flat_lines[header_idx+1:end])
top_data = data[max(1, length(data) - 79):end]
flat_section = join(vcat(header, reverse(top_data)), '\n')

tree_section = join(tree_lines, '\n')

println("\nTop $(length(top_data)) by sample count (proxy for wall-clock time share):")
println(flat_section)
println("\n\n--- Tree view for call-path context ---")
println(tree_section)

summary_path = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
if !isnothing(summary_path)
    open(summary_path, "a") do io
        println(io, "# beer_game() direct CPU profile")
        println(io, "")
        println(io, "One full `beer_game()` call (scenario_count=30, real BlackBoxOptim search, MaxFuncEvals=15000), sampled at the default 1ms interval. Counts are a proxy for wall-clock time share, not exact timings.")
        println(io, "")
        println(io, "## Flat view (top $(length(top_data)) by sample count)")
        println(io, "```")
        println(io, flat_section)
        println(io, "```")
        println(io, "")
        println(io, "## Tree view (call-path context)")
        println(io, "```")
        println(io, tree_section)
        println(io, "```")
    end
end
