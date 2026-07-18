#=
Allocation profile of the actual beer_game() shape (test/policy-beergame-tests.jl):
same network, same real BlackBoxOptim search via optimize!, not a fixed-candidate
proxy - see profile_beergame_direct.jl for why that distinction matters (a CPU
sampler over an isolated minimize! proxy can point at different hotspots than
the real thing).

An earlier version of this script tried to keep the run short instead
(MaxFuncEvals shrunk to 200) so it could sample every single allocation
(sample_rate=1.0). That doesn't work: Profile.Allocs at sample_rate=1.0 is
roughly two orders of magnitude slower than unprofiled execution (measured:
one 200-period scenario simulation, normally ~2s, took 234s), so even 200
trials never finished inside a CI job's lifetime. sample_rate exists
precisely to trade exhaustiveness for tractability on a workload this size -
this profiles the *actual* full beer_game() run (MaxFuncEvals=15000, the
same ~1.4 billion allocations the CPU profile measured) at a low sample
rate instead of shrinking the problem, so the sample is representative of
the real thing rather than a smaller stand-in for it.

Run with:
    julia --project=benchmark benchmark/memory_profile_beergame.jl

Also used by .github/workflows/profile.yml to report results without anyone
needing to run this by hand.
=#

using Random
using Distributions: Poisson
using SupplyChainModeling
using SupplyChainSimulation
using Profile

# Mirrors test/policy-beergame-tests.jl's beer_game() exactly, plus a
# `params` passthrough (not present on the real test function) used only by
# the cheap warmup call below, to compile every code path without paying
# for a real search twice.
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

# Cheap warmup (JIT-compile everything) outside the profiled region.
println("Warming up...")
@time beer_game(scenario_count=2, params=Dict(:MaxFuncEvals => 50.0))

println("Warmup done. Allocation-profiling one real beer_game() call (default scenario_count=30, MaxFuncEvals=15000) at a low sample rate...")

# 1-in-10000: expect on the order of 1.4e9 * 1e-4 ≈ 140000 sampled
# allocations from the real run - plenty for a representative top-N-sites
# report, at a low enough rate that Profile.Allocs' overhead stays close to
# unprofiled execution instead of the ~100x slowdown sample_rate=1.0 has.
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=0.0001 beer_game()

results = Profile.Allocs.fetch()
println("Total sampled allocations: $(length(results.allocs))")

# Aggregate bytes by (allocated type, first non-Profile/Base stack frame).
function first_user_frame(bt)
    for frame in bt
        s = string(frame.file)
        # The innermost frame on every allocation's stacktrace is always the
        # C-side profiler hook itself (gc-alloc-profiler.h), not real Julia
        # code - skip straight past any non-.jl frame instead of trying to
        # exclude specific paths by name.
        if endswith(s, ".jl")
            return frame
        end
    end
    return isempty(bt) ? nothing : bt[1]
end

agg = Dict{Tuple{Type,String}, Tuple{Int,Int}}()  # (type, "func at file:line") => (count, bytes)
for a in results.allocs
    frame = first_user_frame(a.stacktrace)
    key = frame === nothing ? (a.type, "<unknown>") : (a.type, "$(frame.func) at $(frame.file):$(frame.line)")
    c, b = get(agg, key, (0, 0))
    agg[key] = (c + 1, b + a.size)
end

sorted_agg = sort(collect(agg), by = x -> -x[2][2])

table_lines = String[]
push!(table_lines, string(rpad("bytes", 12), rpad("count", 8), rpad("type", 40), "site"))
for (key, (count, bytes)) in sorted_agg[1:min(40, length(sorted_agg))]
    (typ, site) = key
    push!(table_lines, string(rpad(bytes, 12), rpad(count, 8), rpad(string(typ), 40), site))
end
table_section = join(table_lines, '\n')

println("\nTop 40 allocation sites by total bytes (sampled at a 1-in-10000 rate, $(length(results.allocs)) allocs recorded over 1 real beer_game() call):")
println(table_section)

summary_path = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
if !isnothing(summary_path)
    open(summary_path, "a") do io
        println(io, "# beer_game() allocation profile")
        println(io, "")
        println(io, "One real `beer_game()` call (default scenario_count=30, MaxFuncEvals=15000 - the same run the CPU profile measures), sampled at a 1-in-10000 rate. $(length(results.allocs)) allocations sampled.")
        println(io, "")
        println(io, "## Top 40 allocation sites by total bytes")
        println(io, "```")
        println(io, table_section)
        println(io, "```")
    end
end
