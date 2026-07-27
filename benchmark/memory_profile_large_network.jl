#=
Allocation profile of Env construction and a full simulate() call on the
synthetic large network (see large_network.jl) - same rationale as
memory_profile_beergame.jl (sample at a low rate rather than shrinking the
problem, since Profile.Allocs at sample_rate=1.0 is roughly two orders of
magnitude slower than unprofiled execution), and the same two-phase split
as profile_large_network.jl so Env construction's allocations don't drown
out simulate()'s or vice versa.

Run with:
    julia --project=benchmark benchmark/memory_profile_large_network.jl

Network size comes from BENCHMARK_STORAGE_COUNT/BENCHMARK_PRODUCT_COUNT/
BENCHMARK_HORIZON env vars (see large_network.jl), same as benchmarks.jl.

Also runnable via .github/workflows/large-network-benchmark.yml.
=#

using Profile

include("large_network.jl")

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

function print_allocation_profile(title::String, results)
    println("Total sampled allocations: $(length(results.allocs))")

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

    println("\n=== $title ===")
    println("\nTop 40 allocation sites by total bytes ($(length(results.allocs)) allocs sampled):")
    println(table_section)

    summary_path = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
    if !isnothing(summary_path)
        open(summary_path, "a") do io
            println(io, "## $title")
            println(io, "")
            println(io, "$(length(results.allocs)) allocations sampled.")
            println(io, "")
            println(io, "### Top 40 allocation sites by total bytes")
            println(io, "```")
            println(io, table_section)
            println(io, "```")
        end
    end
end

env_only = "--env-only" in ARGS
(; storage_count, product_count, horizon) = large_network_params()
nlanes = 2 * storage_count
sample_rate = 0.0001 # same rate memory_profile_beergame.jl validated as safe

println("Warming up (small network, JIT-compiles Env/State/simulate)...")
@time begin
    warmup_network, warmup_policies = build_large_network(; storage_count=20, product_count=2, horizon=10)
    SupplyChainSimulation.Env(warmup_network, [SupplyChainSimulation.State(warmup_network)], warmup_policies)
    simulate(warmup_network, warmup_policies)
end

println("Warmup done. Building the real network ($storage_count storages x $product_count products x " *
        "$horizon periods, $nlanes lanes) outside the profiled region...")
network, policies = build_large_network(; storage_count, product_count, horizon)
state = SupplyChainSimulation.State(network)

println("\nAllocation-profiling Env construction at a $sample_rate sample rate...")
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=sample_rate SupplyChainSimulation.Env(network, [state], policies)
print_allocation_profile("Env construction allocation profile ($storage_count storages, $nlanes lanes, $product_count products)", Profile.Allocs.fetch())

if env_only
    println("\n--env-only passed: skipping the simulate() allocation profile.")
else
    # Built again here (unprofiled) for the simulate() phase below - Env
    # construction has no randomness of its own, so this is equivalent to
    # the just-profiled call - rather than relying on @profile's return
    # value, which the profiled expression above isn't captured from on
    # purpose.
    env = SupplyChainSimulation.Env(network, [state], policies)

    println("\nAllocation-profiling simulate() at a $sample_rate sample rate...")
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=sample_rate simulate(env, policies, state)
    print_allocation_profile("simulate() allocation profile ($storage_count storages, $nlanes lanes, $product_count products, $horizon periods)", Profile.Allocs.fetch())
end
