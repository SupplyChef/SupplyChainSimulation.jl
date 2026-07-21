#=
Shared synthetic large-network builder, used by benchmarks.jl and by
profile_large_network.jl/memory_profile_large_network.jl - kept in one
place so the three scripts profile/benchmark the exact same network shape
instead of three independently-maintained copies drifting apart.
=#

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

"""
    large_network_params(; small=false)

Reads storage/product/horizon counts from BENCHMARK_STORAGE_COUNT/
BENCHMARK_PRODUCT_COUNT/BENCHMARK_HORIZON env vars, falling back to the
`--small`/default sizing benchmarks.jl has always used. Shared so every
script sizes its network the same way instead of three separate
`parse(Int, get(ENV, ...))` copies.
"""
function large_network_params(; small=false)
    storage_count = parse(Int, get(ENV, "BENCHMARK_STORAGE_COUNT", small ? "20" : "1000"))
    product_count = parse(Int, get(ENV, "BENCHMARK_PRODUCT_COUNT", small ? "2" : "5"))
    horizon = parse(Int, get(ENV, "BENCHMARK_HORIZON", small ? "10" : "52"))
    return (; storage_count, product_count, horizon)
end
