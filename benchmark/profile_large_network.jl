#=
CPU (wall-clock) sampling profile of Env construction and a full simulate()
call on the synthetic large network (see large_network.jl). Unlike
benchmarks.jl's BenchmarkTools timings (aggregate wall-clock/memory only),
this samples the call stack throughout each phase so it can show *where*
within Env construction or simulate() the time actually goes - the two
phases are profiled separately (each with its own Profile.clear()) so one
doesn't drown out the other in the report.

Run with:
    julia --project=benchmark benchmark/profile_large_network.jl

Network size comes from BENCHMARK_STORAGE_COUNT/BENCHMARK_PRODUCT_COUNT/
BENCHMARK_HORIZON env vars (see large_network.jl), same as benchmarks.jl.

Also runnable via .github/workflows/large-network-benchmark.yml.
=#

using Profile

include("large_network.jl")

function print_profile(title::String)
    flat_io = IOBuffer()
    Profile.print(flat_io; format=:flat, sortedby=:count, C=false, mincount=5)
    flat_lines = split(String(take!(flat_io)), '\n')

    tree_io = IOBuffer()
    Profile.print(tree_io; format=:tree, sortedby=:count, C=false, mincount=15, maxdepth=25)
    tree_lines = split(String(take!(tree_io)), '\n')

    # Profile.print(sortedby=:count) sorts rows ascending (lowest count
    # first), so the interesting (highest-count) rows are at the *end* of
    # the output - take the header plus the last 80 data rows, reversed so
    # the highest count reads first (mirrors profile_beergame_direct.jl).
    header_idx = findfirst(l -> occursin("=====", l), flat_lines)
    header = isnothing(header_idx) ? String[] : flat_lines[1:header_idx]
    data = filter(!isempty, isnothing(header_idx) ? flat_lines : flat_lines[header_idx+1:end])
    top_data = data[max(1, length(data) - 79):end]
    flat_section = join(vcat(header, reverse(top_data)), '\n')
    tree_section = join(tree_lines, '\n')

    println("\n=== $title ===")
    println("\nTop $(length(top_data)) by sample count (proxy for wall-clock time share):")
    println(flat_section)
    println("\n\n--- Tree view for call-path context ---")
    println(tree_section)

    summary_path = get(ENV, "GITHUB_STEP_SUMMARY", nothing)
    if !isnothing(summary_path)
        open(summary_path, "a") do io
            println(io, "## $title")
            println(io, "")
            println(io, "### Flat view (top $(length(top_data)) by sample count)")
            println(io, "```")
            println(io, flat_section)
            println(io, "```")
            println(io, "")
            println(io, "### Tree view (call-path context)")
            println(io, "```")
            println(io, tree_section)
            println(io, "```")
        end
    end
end

env_only = "--env-only" in ARGS
(; storage_count, product_count, horizon) = large_network_params()
nlanes = 2 * storage_count

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

println("\nProfiling Env construction...")
Profile.clear()
@profile SupplyChainSimulation.Env(network, [state], policies)
print_profile("Env construction CPU profile ($storage_count storages, $nlanes lanes, $product_count products)")

if env_only
    println("\n--env-only passed: skipping the simulate() profile.")
else
    # Built again here (unprofiled) for the simulate() phase below - Env
    # construction has no randomness of its own, so this is equivalent to
    # the just-profiled call - rather than relying on @profile's return
    # value, which the profiled expression above isn't captured from on
    # purpose.
    env = SupplyChainSimulation.Env(network, [state], policies)

    println("\nProfiling simulate()...")
    Profile.clear()
    @profile simulate(env, policies, state)
    print_profile("simulate() CPU profile ($storage_count storages, $nlanes lanes, $product_count products, $horizon periods)")
end
