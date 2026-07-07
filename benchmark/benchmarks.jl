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
=#

using BenchmarkTools
using Distributions: Poisson
using Random

using SupplyChainModeling
using SupplyChainSimulation

"""
    build_large_network(; storage_count=1000, product_count=5, horizon=52, seed=42)

Builds a synthetic supplier -> many-storages -> many-customers network sized
for performance benchmarking (`storage_count * product_count` storage/product
combinations, plus one customer and one sell lane per storage).
"""
function build_large_network(; storage_count=1000, product_count=5, horizon=52, seed=42)
    Random.seed!(seed)

    network = SupplyChain(horizon)

    products = [Product("p$i") for i in 1:product_count]
    for p in products
        add_product!(network, p)
    end

    supplier = Supplier("supplier")
    add_supplier!(network, supplier)
    for p in products
        add_product!(supplier, p; unit_cost=1.0)
    end

    policies = Dict{Tuple{Lane, Product}, InventoryOrderingPolicy}()

    for i in 1:storage_count
        storage = Storage("storage$i")
        add_storage!(network, storage)

        customer = Customer("customer$i")
        add_customer!(network, customer)

        sell_lane = Lane(storage, customer; unit_cost=0.1)
        add_lane!(network, sell_lane)

        supply_lane = Lane(supplier, storage; unit_cost=1.0, time=2)
        add_lane!(network, supply_lane)

        for p in products
            add_product!(storage, p; unit_holding_cost=0.1)

            demand = Float64.(rand(Poisson(10), horizon))
            add_demand!(network, customer, p, demand; sales_price=2.0, lost_sales_cost=1.0)

            policies[(supply_lane, p)] = NetSSOrderingPolicy(20, 60)
        end
    end

    return network, policies
end

function main()
    small = "--small" in ARGS
    storage_count = small ? 20 : 1000
    product_count = small ? 2 : 5
    horizon = small ? 10 : 52

    println("Building network: $storage_count storages x $product_count products x $horizon periods " *
            "($(storage_count * product_count) storage/product combinations)...")
    network, policies = build_large_network(; storage_count, product_count, horizon)

    println("\nBenchmarking Env construction (isolates per-lane-per-period Trip materialization)...")
    env_bench = @benchmark SupplyChainSimulation.Env($network, [SupplyChainSimulation.State($network)], $policies) samples=5 seconds=180
    display(env_bench)
    println()

    println("\nBenchmarking full simulate() call...")
    simulate_bench = @benchmark simulate($network, $policies) samples=3 seconds=300
    display(simulate_bench)
    println()
end

main()
