#=
CPU (wall-clock) sampling profile of optimize!'s hot loop (minimize!) under
the beer game test's exact network shape (see test/policy-beergame-tests.jl).
Prints a flat view (by sample count, a proxy for wall-clock time share) and a
tree view for call-path context.

Run with:
    julia --project=benchmark benchmark/profile_beergame.jl

Also used by .github/workflows/profile.yml to report results without anyone
needing to run this by hand.
=#

using Random
using Distributions: Poisson
using SupplyChainModeling
using SupplyChainSimulation
using Profile

Random.seed!(3)

# --- Reconstruct beer_game()'s network exactly ---
scenario_count = 30
horizon = 200

product = SupplyChainModeling.Product("product")

customer = Customer("customer")
retailer = Storage("retailer")
add_product!(retailer, product; unit_holding_cost=0.1, initial_inventory=20)
wholesaler = Storage("wholesaler")
add_product!(wholesaler, product; unit_holding_cost=0.1, initial_inventory=20)
factory = Storage("factory")
add_product!(factory, product; unit_holding_cost=0.1, initial_inventory=20)
supplier = Supplier("supplier")

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

supplychains = [n() for i in 1:scenario_count]

lane_policies = Dict((l2, product) => policy2,
                      (l3, product) => policy3,
                      (l4, product) => policy4)

# --- Mirror optimize!'s prep phase (Optimization.jl) exactly ---
initial_states = SupplyChainSimulation.State.(supplychains)
envs = [SupplyChainSimulation.Env(supplychain, initial_states, lane_policies; record_history=false) for supplychain in supplychains]

sorted_keys = sort(collect(keys(lane_policies)); by = k -> (string(k[1]), k[2].name))
policies = unique([lane_policies[k] for k in sorted_keys])
x0 = convert(Array{Float64,1}, vcat([SupplyChainSimulation.get_parameters(policy) for policy in policies]...))

candidates = [x0,
              x0 .+ 1.0,
              max.(0.0, x0 .- 0.5),
              x0 .* 2.0,
              rand(length(x0)) .* 5000.0]

function run_trials(n)
    for i in 1:n
        SupplyChainSimulation.minimize!(lane_policies, policies, envs, initial_states,
                                         candidates[mod1(i, length(candidates))];
                                         cost_function=metrics_cost_function)
    end
end

# Warm up (JIT-compile everything) outside the profiled region.
run_trials(20)

println("Warmup done. Profiling (this runs 300 minimize! calls under the sampling profiler)...")

Profile.clear()
# 1000 microsecond (1ms) sample interval is the default; leave it be unless
# the run is too short to get enough samples.
@time @profile run_trials(300)

flat_io = IOBuffer()
Profile.print(flat_io; format=:flat, sortedby=:count, C=false, mincount=5)
flat_lines = split(String(take!(flat_io)), '\n')

tree_io = IOBuffer()
Profile.print(tree_io; format=:tree, sortedby=:count, C=false, mincount=15, maxdepth=25)
tree_lines = split(String(take!(tree_io)), '\n')

flat_section = join(flat_lines[1:min(80, length(flat_lines))], '\n')
tree_section = join(tree_lines[1:min(150, length(tree_lines))], '\n')

println("\nTop 80 by sample count (proxy for wall-clock time share):")
println(flat_section)
println("\n\n--- Tree view truncated to depth-relevant frames for call-path context ---")
println(tree_section)

summary_path = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
if !isnothing(summary_path)
    open(summary_path, "a") do io
        println(io, "# beer_game() CPU profile")
        println(io, "")
        println(io, "300 `minimize!` calls, sampled at the default 1ms interval. Counts are a proxy for wall-clock time share, not exact timings.")
        println(io, "")
        println(io, "## Flat view (top 80 by sample count)")
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
