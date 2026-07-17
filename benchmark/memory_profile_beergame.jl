#=
Allocation profile of the actual beer_game() shape (test/policy-beergame-tests.jl):
same network, same real BlackBoxOptim search via optimize!, not a fixed-candidate
proxy - see profile_beergame_direct.jl for why that distinction matters (a CPU
sampler over an isolated minimize! proxy can point at different hotspots than
the real thing).

Sampling every allocation (sample_rate=1.0) across a *full* beer_game() run
(MaxFuncEvals=15000) is infeasible - that run makes on the order of 1.4
*billion* allocations (see the CPU profile's `@time` output), far more than
Profile.Allocs can record. Instead this profiles one real beer_game() call
with a much smaller MaxFuncEvals (via a `params` passthrough not present on
the real test function): same code paths, same relative allocation *shape*
per trial, just far fewer trials - full-rate sampling stays accurate and
tractable.

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
# `params` passthrough (not present on the real test function) so this can
# run a real-shaped but much shorter optimize! search.
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

println("Warmup done. Allocation-profiling one beer_game() call (scenario_count=30, MaxFuncEvals=200 - real search shape, short enough to sample every allocation)...")

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 beer_game(; params=Dict(:MaxFuncEvals => 200.0))

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

println("\nTop 40 allocation sites by total bytes (sampled, $(length(results.allocs)) allocs over 1 beer_game(MaxFuncEvals=200) call):")
println(table_section)

summary_path = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
if !isnothing(summary_path)
    open(summary_path, "a") do io
        println(io, "# beer_game() allocation profile")
        println(io, "")
        println(io, "One `beer_game(MaxFuncEvals=200)` call (real search shape, shortened so `sample_rate=1.0` can record every allocation). $(length(results.allocs)) allocations sampled.")
        println(io, "")
        println(io, "## Top 40 allocation sites by total bytes")
        println(io, "```")
        println(io, table_section)
        println(io, "```")
    end
end
