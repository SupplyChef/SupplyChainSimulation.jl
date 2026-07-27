#=
Performance/scale benchmark suite for SupplyChainSimulation.jl.

Targets the bottlenecks already identified: eager per-lane-per-period `Trip`
materialization in `Env` construction, and the fully serial `simulate` loop.
Sized to roughly Fortune-100 scale (thousands of storage/product combinations)
rather than the small networks used in the correctness test suite.

Run with:
    julia --project=benchmark benchmark/benchmarks.jl

For a quick local smoke test with a much smaller network, pass a scale factor:
    julia --project=benchmark benchmark/benchmarks.jl --small

Network size is otherwise overridable via BENCHMARK_STORAGE_COUNT/
BENCHMARK_PRODUCT_COUNT/BENCHMARK_HORIZON env vars, e.g. to probe a
specific scale:
    BENCHMARK_STORAGE_COUNT=10000 julia --project=benchmark benchmark/benchmarks.jl

Pass --env-only to skip the full simulate() benchmark and only measure
Env construction - useful at large storage counts, where simulate()'s
runtime (dominated by the O(horizon) simulate() loop itself, not by
anything Env construction isolates) would otherwise dominate the wall
time of a run that's really trying to isolate Env construction's own
O(lanes x horizon)/O(lanes x products) costs.

See profile_large_network.jl/memory_profile_large_network.jl for CPU/
allocation sampling profiles of the same network - this script only
reports aggregate timing/memory via BenchmarkTools, not where within
Env construction or simulate() the time/allocations actually go.
=#

using BenchmarkTools

include("large_network.jl")

function main()
    small = "--small" in ARGS
    env_only = "--env-only" in ARGS
    (; storage_count, product_count, horizon) = large_network_params(; small)
    nlanes = 2 * storage_count # one sell_lane + one supply_lane per storage, see build_large_network

    println("Building network: $storage_count storages x $product_count products x $horizon periods " *
            "($nlanes lanes, $(storage_count * product_count) storage/product combinations, " *
            "$(nlanes * product_count) lane/product combinations)...")
    network, policies = build_large_network(; storage_count, product_count, horizon)

    println("\nBenchmarking Env construction (isolates per-lane-per-period Trip materialization, " *
            "and the O(lanes x products) get_lane_policies/past_orders_buffers costs)...")
    env_bench = @benchmark SupplyChainSimulation.Env($network, [SupplyChainSimulation.State($network)], $policies) samples=5 seconds=180
    display(env_bench)
    println()

    if env_only
        println("\n--env-only passed: skipping the full simulate() benchmark.")
        return
    end

    println("\nBenchmarking full simulate() call...")
    simulate_bench = @benchmark simulate($network, $policies) samples=3 seconds=300
    display(simulate_bench)
    println()
end

main()
