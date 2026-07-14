# PERFORMANCE PROPOSAL: Order of Magnitude Speedups for SupplyChainSimulation.jl

## 1. Executive Summary
This proposal outlines a concrete architectural redesign of `SupplyChainSimulation.jl` to increase execution speed by **at least an order of magnitude (10x–100x+)** without utilizing parallelism, while establishing a direct path to massive **multi-core CPU parallelization** and **GPU acceleration**.

The current implementation relies on a pointer-heavy, dictionary-based, and heap-allocated object model. By transitioning the core simulation state and logic to a **flat, cache-friendly, contiguous layout**—where nodes, products, and lanes are identified by contiguous integer indices and simulated via dense arrays/matrices—we eliminate dynamic dispatch, minimize GC overhead, and maximize memory throughput.

This document analyzes the current bottlenecks, details the flat memory architecture, specifies how each simulation step translates to high-performance kernels, maps out parallelism/GPU implementation, and outlines a **step-by-step, working-to-working evolutionary plan** to refactor the package while maintaining full API compatibility and passing all existing tests.

---

## 2. Current Bottlenecks & Profiling Analysis

As demonstrated by flat profile traces and code inspection, the existing simulator spends most of its time performing:
1. **Dictionary Lookups & Hashing:** Operations like `get(state.on_hand_totals, (to, product), 0)` and accessing `state.pending_outbound_order_lines[(location, product)]` require hashing compound keys (e.g., tuples of custom types `(Storage, Product)`). This causes heavy CPU cache misses and overhead in `hashindex` and `ht_keyindex`.
2. **Heap Allocations and GC Overhead:** Every simulated period triggers significant garbage collection (GC) activity due to:
   - Constructing short-lived `OrderLine` objects and pushing/filtering them in vectors like `OrderLine[]`.
   - Modifying and copying `Dict` structures for `on_hand_inventory`, `in_transit_inventory`, and pending orders.
   - Pushing items into `Set{OrderLine}` and `Set{Trip}`.
3. **Dynamic Dispatch & Type Instability:** Storing heterogeneous nodes under abstract fields (e.g., `Node` or `UnionAll` types) forces the Julia compiler to generate slow, dynamically-dispatched lookups for node properties instead of devirtualized, inlined machine instructions.
4. **Indirection & Cache Unfriendliness:** The storage of variables across scattered objects (pointer-chasing) means data is not spatially contiguous. The CPU cannot leverage SIMD vectorization or prefetching effectively.

---

## 3. Highly Optimized Flat Memory Architecture

To achieve extreme performance, we must represent the simulation state as flat, statically typed, contiguous matrices.

### 3.1 Index Mapping
At the start of `simulate` or `optimize!`, we map all objects to contiguous, 1-based integer indices:
* **Locations:** $L$ locations mapped to integers $1 \dots L$. We can partition this range so that:
  - Suppliers: $1 \dots N_{\text{sup}}$
  - Storages: $(N_{\text{sup}} + 1) \dots (N_{\text{sup}} + N_{\text{store}})$
  - Customers: $(N_{\text{sup}} + N_{\text{store}} + 1) \dots L$
* **Products:** $P$ products mapped to integers $1 \dots P$.
* **Lanes:** $K$ lanes mapped to integers $1 \dots K$.

### 3.2 Flat Simulation State (`FlatState`)
The entire `State` struct is replaced internally with pre-allocated, flat arrays:
* **On-Hand Inventory:** A 3D dense array of shape `(L, P, MaxAge)`. `on_hand[l, p, a]` holds the inventory of product `p` at location `l` with age `a`.
  - Alternatively, if expiration is rarely used, a 2D dense matrix `on_hand_totals[l, p]` of shape `(L, P)` can be maintained, and ages are only tracked when `max_age` is finite.
* **In-Transit Inventory:** A 3D dense array `in_transit[l, p, t]` of shape `(L, P, Horizon)` tracking arrivals per location, product, and time slot.
* **Pending Orders Matrix:** A 3D dense array `pending_orders[origin, dest, p]` of shape `(L, L, P)` representing the total quantity on backorder.
* **Order Line Queue:** Instead of heap-allocated `OrderLine` structs, we use pre-allocated flat vectors (or circular buffers) of primitive types:
  - `ol_creation_time::Vector{Int32}`
  - `ol_origin::Vector{Int32}`
  - `ol_destination::Vector{Int32}`
  - `ol_product::Vector{Int32}`
  - `ol_quantity::Vector{Int32}`
  - `ol_due_date::Vector{Int32}`
  - `ol_trip::Vector{Int32}`
  - Active/free lists of indices are used to avoid allocating/deallocating elements.
* **SimMetrics:** Kept as primitive `Float64` accumulator fields directly inside the flat state (e.g. `metrics_sales`, `metrics_lost_sales`, `metrics_holding_costs`).

### 3.3 Flat Environment (`FlatEnv`)
The `Env` configuration is flattened into lookup vectors:
* **Lane Properties:** Flat vectors of length $K$ for `lane_origin[k]`, `lane_destination[k]`, `lane_unit_cost[k]`, `lane_lead_time[k]`, and `lane_minimum_quantity[k]`.
* **Holding Costs:** Matrix `unit_holding_cost[l, p]` of shape `(L, P)`.
* **Storage Capacities:** Matrix `max_capacity[l, p]` of shape `(L, P)`.
* **Sales and Demand Configuration:**
  - `demand_price[l, p]` and `lost_sales_cost[l, p]` of shape `(L, P)`.
  - `demand_series[l, p, t]` of shape `(L, P, Horizon)` representing the demand at time `t` for customer `l` and product `p`.

---

## 4. Mapping Simulation Steps to High-Performance Kernels

By leveraging flat arrays, each simulation step becomes a tight, cache-friendly loop over contiguous memory, enabling the compiler to apply SIMD auto-vectorization.

### 4.1 Receive Inventory Kernel
For a given period `time`, we iterate through all storages `l` and products `p`:
```julia
# Contiguous memory iteration over p and l
for l in storages
    for p in 1:P
        quantity = in_transit[l, p, time]
        if quantity > 0
            in_transit[l, p, time] = 0
            max_cap = max_capacity[l, p]
            if isinf(max_cap)
                # Add to age 1 (or current time)
                on_hand[l, p, 1] += quantity
                on_hand_totals[l, p] += quantity
            else
                on_hand_now = on_hand_totals[l, p]
                accepted = min(quantity, max(0, Int(max_cap) - on_hand_now))
                overflow = quantity - accepted

                on_hand[l, p, 1] += accepted
                on_hand_totals[l, p] += accepted

                if overflow > 0
                    metrics_overflow_costs[1] += overflow * overflow_cost[l, p]
                    if time < horizon
                        in_transit[l, p, time + 1] += overflow
                    end
                end
            end
        end
    end
end
```
* **Performance gain:** No tuple keys, no dictionary lookups. This compiles to sequential memory writes, utilizing CPU cache prefetching.

### 4.2 Place & Receive Orders Kernel
During ordering, locations are iterated in **reverse topological order** (from downstream customers up to upstream suppliers).
Instead of storing and allocating individual `OrderLine` objects:
```julia
# Customer demand placement
for l in customers
    for p in 1:P
        qty = demand_series[l, p, time]
        if qty > 0
            # Instantly place in a flat backorder queue
            backorders[customer_parent_storage[l], l, p] += qty
            metrics_orders[1] += qty
            metrics_demand[1] += qty * sales_price[l, p]
        end
    end
end

# Replenishment policy evaluation
for l in storages_reversed
    for p in 1:P
        # Compute net inventory using fast array math
        net_inv = on_hand_totals[l, p] +
                  sum_in_transit(in_transit, l, p, time, horizon) +
                  inbound_orders[l, p] - outbound_orders[l, p]

        # Policy is evaluated as a simple inline function branch
        qty = evaluate_policy(l, p, net_inv, time)
        if qty > 0
            # Place order on upstream supplier
            supplier_id = lane_origin[replenishment_lane[l, p]]
            backorders[supplier_id, l, p] += qty
            metrics_orders[1] += qty
        end
    end
end
```

### 4.3 Send Inventory Kernel
Processing backorders and shipping units:
```julia
for l in storages_and_suppliers
    for p in 1:P
        # Find pending orders bound for downstream
        for dest in downstream_of[l]
            qty_due = backorders[l, dest, p]
            if qty_due > 0
                available = on_hand_totals[l, p]
                to_ship = min(qty_due, available)

                if to_ship > 0
                    lead = lead_time[l, dest]
                    if time + lead <= horizon
                        in_transit[dest, p, time + lead] += to_ship
                    end
                    if !is_supplier[l]
                        remove_on_hand!(on_hand, l, p, to_ship)
                        on_hand_totals[l, p] -= to_ship
                    end
                    backorders[l, dest, p] -= to_ship

                    # Record metric costs directly
                    metrics_unit_costs[1] += unit_shipping_cost[l, dest] * to_ship
                    metrics_sales[1] += to_ship * sales_price[dest, p]
                end
            end
        end
    end
end
```
* **Performance gain:** Completely removes the sorting and filtering of `OrderLine` vectors. The shipment is modeled as a direct transfer between two integer indices in flat matrices.

---

## 5. Parallelism and GPU Execution Path

A flat contiguous memory layout is highly conducive to parallel architectures.

### 5.1 Multi-Threaded Parallelism (CPU Cores)
Because policies are optimized across thousands of evaluations (e.g. `bboptimize` evaluating 15,000 function runs) and many parallel initial scenarios (e.g., Monte Carlo simulations in the Beer Game), we can achieve linear speedups with CPU multi-threading:
* **Scenario-level Parallelism:** Distribute individual scenario runs within `minimize!` across a `Threads.@threads` loop. Since each thread operates on its own independent `FlatState` instance, there are no race conditions or synchronization overheads.
* **SIMD Vectorization:** Inside the single-threaded simulation loop, loops over the product dimension `for p in 1:P` can be annotated with `@simd` or `@views` to execute instructions in parallel on the CPU register level.

### 5.2 GPU Acceleration
A GPU requires flat, statically-sized global memory and avoids branch-heavy dynamic code. Transitioning to `FlatState` allows a 1-to-1 port to CUDA/Metal kernels:
* **Memory Allocation:** All state variables (`on_hand`, `in_transit`, `backorders`) are pre-allocated as `CuArray`s on the GPU.
* **Kernel Execution:**
  - Each thread on the GPU simulates a single **(Scenario, Location, Product)** tuple.
  - At each simulation timestep, a series of GPU synchronization boundaries (using `CUDA.@sync` or parallel reduction kernels) execute the Receive, Order, Ship, and Expire steps.
  - Since memory accesses are contiguous along the Product and Location dimensions, memory coalescing is maximized, unlocking massive throughput.

---

## 6. Step-by-Step "Working-to-Working" Evolution Plan

To ensure all correctness tests pass at every step of development, we will adopt a phased, backward-compatible evolution strategy:

### Phase 1: Establish High-Performance Interfaces Side-by-Side
1. Maintain the existing `State`, `Env`, and `simulate` code exactly as they are today.
2. Build an internal module `FlatSimulation` that implements `FlatState`, `FlatEnv`, and `flat_simulate`.
3. Write bidirectional adapter functions:
   - `to_flat_env(env::Env, policies)::FlatEnv`
   - `to_flat_state(state::State)::FlatState`
   - `update_legacy_state!(legacy::State, flat::FlatState)`
4. Set up verification tests that run both `simulate` and `flat_simulate` on identical inputs, validating that they produce bit-identical results.

### Phase 2: Integrate `FlatSimulation` into the Optimization Loop
1. Modify `optimize!` to translate its legacy input networks into `FlatEnv` and `FlatState`.
2. Execute the `bboptimize` trials entirely using the high-performance `flat_simulate` kernel.
3. Because trial simulations represent 99% of `optimize!` computation time, this yields an immediate 10x–50x speedup for policy optimization, while keeping the user-facing inputs and outputs perfectly backward-compatible.
4. Run all calibration, beer game, and policy optimization tests to confirm correctness and baseline convergence.

### Phase 3: Transition the Public API to Flat Internals
1. Refactor the public `State` and `Env` structs to wrap `FlatState` and `FlatEnv` internally.
2. Implement deprecated/legacy properties as on-the-fly computed views or getters:
   - For example, if a legacy test reads `state.on_hand_inventory`, we implement it as a custom property/method that builds the dictionary from `FlatState`'s dense array on demand.
3. Fully replace the serial `simulate` with `flat_simulate`, making the high-performance execution path the default for all simulations.
4. Run the entire test suite (`runtests.jl`) to ensure absolute API compatibility and correctness.

### Phase 4: Implement Parallelism and GPU Extensions
1. Introduce multi-threaded evaluations in `optimize!` via `Threads.@threads`.
2. Write a `gpu_simulate` function leveraging `CUDA.jl` for users simulating massive networks or running large-scale training workloads (e.g., reinforcement learning).
3. Add a benchmark suite to track execution speedups, aiming to consistently outperform the previous architecture by orders of magnitude.
