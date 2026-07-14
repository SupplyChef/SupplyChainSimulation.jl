# PERFORMANCE PROPOSAL: Order of Magnitude Speedups for SupplyChainSimulation.jl

## 1. Executive Summary
This proposal outlines a concrete architectural redesign of `SupplyChainSimulation.jl` to increase execution speed by **at least an order of magnitude (10x–100x+)** without utilizing parallelism, while establishing a direct path to massive **multi-core CPU parallelization** and **GPU acceleration**.

The current implementation relies on a pointer-heavy, dictionary-based, and heap-allocated object model. By transitioning the core simulation state and logic to a **flat, cache-friendly, contiguous layout**—where nodes, products, and lanes are identified by contiguous integer indices and simulated via dense arrays/matrices—we eliminate dynamic dispatch, minimize GC overhead, and maximize memory throughput.

This document analyzes the current bottlenecks, details the flat memory architecture, specifies how each simulation step translates to high-performance kernels, maps out parallelism/GPU implementation, addresses memory footprint concerns at scale with **index compression**, outlines a **step-by-step, in-place, bit-by-bit evolutionary plan**, and provides a **Senior Developer Critique** highlighting potential implementation pitfalls and architectural improvements.

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

## 4. Mitigating Memory Blow-up: Index Compression & Sparse Flat Structures

### 4.1 The Memory Risk
At scale, a naive dense multidimensional representation introduces substantial memory overhead:
* If $L = 10,000$ and $P = 1,000$, a 2D matrix like `on_hand_totals[L, P]` requires $10,000 \times 1,000 \times 8 \text{ bytes} \approx 80 \text{ MB}$. This is perfectly manageable.
* However, tracking age with a naive 3D array `on_hand[L, P, MaxAge]` where $MaxAge = 100$ periods scales to $10,000 \times 1,000 \times 100 \times 8 \text{ bytes} \approx 8 \text{ GB}$.
* Even worse, a naive 3D pending order matrix `pending_orders[L, L, P]` scales to $10,000^2 \times 1,000 \times 8 \text{ bytes} \approx 800 \text{ GB}$!

In real-world networks, supply chains are highly **sparse**:
1. **Product Coexistence:** A single storage location never holds all $1,000$ products. Usually, each storage holds a small subset of products (e.g., $10$ to $50$ products).
2. **Network Connectivity:** A location is only connected to a few adjacent upstream and downstream locations (each location connects to $\approx 1$ to $5$ lanes, not all $10,000$).
3. **Active Backorders:** At any given timestep, backorders are only active on valid Lanes for specific Products, representing a fraction of the $L \times L \times P$ space.

### 4.2 Solution: Index Compression (The Active Pair Approach)
Instead of indexing globally by `(Location, Product)` or `(Location, Location, Product)`, we compress our state arrays to only allocate memory for **active, valid pairs**:

#### A. Product-Location (PL) Compression
We define an active set of valid product-location combinations $PL_{\text{valid}} = \{(l, p) \mid \text{product } p \text{ is registered at location } l\}$.
* Let $M$ be the total number of valid $(l, p)$ pairs. At scale, $M \ll L \times P$. For example, if each of the $10,000$ locations holds an average of $20$ products, $M = 200,000$ valid combinations (instead of the $10,000,000$ dense elements).
* We map each valid $(l, p)$ combination to a single contiguous integer index $id \in 1 \dots M$.
* We maintain two lightweight, flat lookup arrays:
  - `pl_location::Vector{Int32}` of length $M$
  - `pl_product::Vector{Int32}` of length $M$
* **On-Hand Inventory** is represented as a highly compressed 2D array: `on_hand[id, age]` of shape `(M, MaxAge)`.
  - For $M = 200,000$ and $MaxAge = 100$, this uses $200,000 \times 100 \times 8 \text{ bytes} \approx 160 \text{ MB}$, a massive **50x reduction** from $8 \text{ GB}$!

#### B. Lane-Product (LP) Compression for Backorders and Ships
Instead of an $L \times L \times P$ matrix for pending orders, we index backorders by Lanes. Since shipments can only occur over valid physical lanes, pending orders can only exist where there is a Lane.
* Let $K$ be the number of lanes. If $K = 20,000$ and the average lane carries $5$ products, we have $N_{\text{lp}} = 100,000$ active `(Lane, Product)` combinations.
* We map each active `(Lane, Product)` to a contiguous index $id_{\text{lp}} \in 1 \dots N_{\text{lp}}$.
* We represent pending orders/backorders as a 1D vector: `pending_orders::Vector{Int64}` of length $N_{\text{lp}}$.
  - For $N_{\text{lp}} = 100,000$, this takes $100,000 \times 8 \text{ bytes} \approx 800 \text{ KB}$, a colossal **1,000,000x reduction** from the $800 \text{ GB}$ dense matrix!

#### C. In-Transit Inventory Compression
Similarly, `in_transit` inventory only arrives via a Lane for a registered Product.
* We represent in-transit quantities as a 2D array: `in_transit[id_lp, horizon]` of shape `(N_lp, Horizon)`.
  - For $N_{\text{lp}} = 100,000$ and $Horizon = 100$, this takes $100,000 \times 100 \times 8 \text{ bytes} \approx 80 \text{ MB}$.

### 4.3 Preserving Cache-Friendliness, SIMD, and GPU Compatibility
Because `on_hand`, `in_transit`, and `pending_orders` are still mapped to contiguous integer indices ($1 \dots M$ and $1 \dots N_{\text{lp}}$) and stored as flat 1D/2D dense arrays, we completely preserve performance:
* **Vectorization:** CPU registers can still stream through `on_hand[id, age]` or `in_transit[id_lp, t]` contiguously.
* **No Pointer Chasing / Hashing:** To find the on-hand inventory of a compressed pair during the simulation, we use flat lookup mappings or index offsets.
* **GPU Memory Friendliness:** GPUs excel at performing parallel operations on flat vectors of size $M$ or $N_{\text{lp}}$. The memory footprint fits entirely within standard GPU VRAM (megabytes instead of gigabytes), avoiding costly device-to-host memory thrashing.

---

## 5. Mapping Simulation Steps to High-Performance Kernels

By leveraging flat, compressed arrays, each simulation step is mapped to tight, cache-friendly loops over contiguous memory.

### 5.1 Receive Inventory Kernel
For a given period `time`, we iterate through all active Product-Location indices $id \in 1 \dots M$ where the location is a Storage:
```julia
# Fast, sequential memory scan over compressed product-location pairs
for id in 1:M
    l = pl_location[id]
    if is_storage[l]
        p = pl_product[id]
        # Map to Lane-Product id to retrieve arrivals
        lp_id = pl_to_lp[id]
        quantity = in_transit[lp_id, time]

        if quantity > 0
            in_transit[lp_id, time] = 0
            max_cap = max_capacity[id]
            if isinf(max_cap)
                on_hand[id, 1] += quantity
                on_hand_totals[id] += quantity
            else
                on_hand_now = on_hand_totals[id]
                accepted = min(quantity, max(0, Int(max_cap) - on_hand_now))
                overflow = quantity - accepted

                on_hand[id, 1] += accepted
                on_hand_totals[id] += accepted

                if overflow > 0
                    metrics_overflow_costs[1] += overflow * overflow_cost[id]
                    if time < horizon
                        in_transit[lp_id, time + 1] += overflow
                    end
                end
            end
        end
    end
end
```

### 5.2 Place & Receive Orders Kernel
During ordering, locations are iterated in reverse topological order.
```julia
# Customer demand placement
for id in 1:M
    l = pl_location[id]
    if is_customer[l]
        p = pl_product[id]
        qty = demand_series[id, time]
        if qty > 0
            # Instantly place in a flat backorder array at the corresponding LP index
            lp_id = pl_to_customer_lp[id]
            pending_orders[lp_id] += qty
            metrics_orders[1] += qty
            metrics_demand[1] += qty * sales_price[id]
        end
    end
end

# Replenishment policy evaluation
for l in storages_reversed
    for p in products_at_location[l]
        id = pl_index_mapping[l, p]
        lp_id = pl_to_lp[id]

        # Compute net inventory using fast flat array math
        net_inv = on_hand_totals[id] +
                  sum_in_transit_slice(in_transit, lp_id, time, horizon) +
                  inbound_orders[id] - outbound_orders[id]

        # Policy is evaluated as an inlined function branch
        qty = evaluate_policy(policy_id[id], net_inv, time)
        if qty > 0
            pending_orders[lp_id] += qty
            metrics_orders[1] += qty
        end
    end
end
```

### 5.3 Send Inventory Kernel
Processing backorders and shipping units:
```julia
for lp_id in 1:N_lp
    qty_due = pending_orders[lp_id]
    if qty_due > 0
        origin_id = lp_origin_pl_id[lp_id]
        dest_id = lp_dest_pl_id[lp_id]

        available = is_supplier[lp_origin[lp_id]] ? qty_due : on_hand_totals[origin_id]
        to_ship = min(qty_due, available)

        if to_ship > 0
            lead = lane_lead_time[lp_lane[lp_id]]
            if time + lead <= horizon
                in_transit[lp_id, time + lead] += to_ship
            end
            if !is_supplier[lp_origin[lp_id]]
                remove_on_hand!(on_hand, origin_id, to_ship)
                on_hand_totals[origin_id] -= to_ship
            end
            pending_orders[lp_id] -= to_ship

            # Record metric costs directly
            metrics_unit_costs[1] += lane_unit_cost[lp_lane[lp_id]] * to_ship
            if is_customer[lp_dest[lp_id]]
                metrics_sales[1] += to_ship * sales_price[dest_id]
            end
        end
    end
end
```

---

## 6. Parallelism and GPU Execution Path

### 6.1 Multi-Threaded Parallelism (CPU Cores)
Because policies are optimized across thousands of evaluations (e.g. `bboptimize` evaluating 15,000 function runs) and many parallel initial scenarios (e.g., Monte Carlo simulations in the Beer Game), we can achieve linear speedups with CPU multi-threading:
* **Scenario-level Parallelism:** Distribute individual scenario runs within `minimize!` across a `Threads.@threads` loop. Since each thread operates on its own independent `FlatState` instance, there are no race conditions or synchronization overheads.
* **SIMD Vectorization:** Inside the single-threaded simulation loop, loops over the product dimension `for p in 1:P` can be annotated with `@simd` or `@views` to execute instructions in parallel on the CPU register level.

### 6.2 GPU Acceleration
A GPU requires flat, statically-sized global memory and avoids branch-heavy dynamic code. Transitioning to `FlatState` allows a 1-to-1 port to CUDA/Metal kernels:
* **Memory Allocation:** All state variables (`on_hand`, `in_transit`, `backorders`) are pre-allocated as `CuArray`s on the GPU.
* **Kernel Execution:**
  - Each thread on the GPU simulates a single **(Scenario, Location, Product)** tuple.
  - At each simulation timestep, a series of GPU synchronization boundaries (using `CUDA.@sync` or parallel reduction kernels) execute the Receive, Order, Ship, and Expire steps.
  - Since memory accesses are contiguous along the Product and Location dimensions, memory coalescing is maximized, unlocking massive throughput.

---

## 7. Step-by-Step "Working-to-Working" In-Place Evolution Plan

Instead of building a separate high-performance implementation block on the side (which can lead to code duplication, design drift, and massive integration headaches), we can evolve the **existing architecture bit-by-bit in-place**.

At each small, incremental step, we introduce a flat layout for a specific subset of the simulation state, map the legacy dictionaries to read/write from this flat memory via custom getters/setters, and verify that 100% of the correctness tests continue to pass.

### Step 1: Pre-calculate Compressed Mappings in `Env`
* **Action:** When `Env` is constructed, build the high-performance flat lookup structures:
  - Establish a stable order and construct contiguous integer arrays mapping every active product-location to an ID: `pl_location::Vector{Int32}`, `pl_product::Vector{Int32}`.
  - Construct the lookup tables: `pl_to_lp`, `lane_lead_time`, `max_capacity`, and `sales_price` arrays.
* **In-Place Integration:** At this step, the core simulation continues to run on legacy dictionaries. We simply build these pre-calculated indices in `Env` and store them in inactive fields.
* **Status:** Working to working. No behavior changes. Correctness tests pass.

### Step 2: Refactor `on_hand_totals` to Flat Memory in `State`
* **Action:** Replace the `state.on_hand_totals` dictionary in `State` with a flat `Vector{Int64}` of size $M$ (where $M$ is the number of active Product-Location pairs pre-calculated in Step 1).
* **Bridging & Legacy Support:**
  - Rewrite `get_on_hand_inventory(state, location, product)` to look up the pre-calculated integer ID for `(location, product)` and directly index into the flat `Vector`.
  - Maintain backward compatibility by providing custom overload/view methods so that any old code accessing `state.on_hand_totals` directly (like tests or custom policies) receives a virtual dictionary view (using a lightweight wrapper implementing `AbstractDict`) that reads/writes to the flat array.
* **Status:** Working to working. Dynamic dictionary hashing is eliminated for on-hand inventory totals. Core correctness tests pass.

### Step 3: Flatten `in_transit_inventory` and `overflow_inventory`
* **Action:**
  - Replace the dictionary `state.in_transit_inventory` with a 2D dense `Matrix{Int64}` of size $(N_{\text{lp}}, Horizon)$.
  - Replace `state.overflow_inventory` with a compressed vector/matrix.
* **Bridging & Legacy Support:**
  - Rewrite `get_in_transit_inventory` and `add_in_transit_inventory!` to look up the compressed `lp_id` of the lane-product and perform direct matrix indexing.
  - Expose a virtual dictionary view wrapper for any legacy accesses to `state.in_transit_inventory`.
* **Status:** Working to working. In-transit lookups now run in $O(1)$ time with zero heap allocations. Tests pass.

### Step 4: Refactor the Simulation Loop Steps
* **Action:** Convert individual parts of the `simulate` loop inside `Simulation.jl` to use flat array operations rather than legacy dict lookups:
  - Migrate `receive_inventory!` to run sequentially across the compressed active Product-Location pairs.
  - Migrate policy evaluation inside `place_orders` to loop through the pre-sorted list of active locations and products, executing policies directly on flat inventories.
* **Status:** Working to working. Performance increases significantly as we eliminate sequential dictionary lookups during the main loop steps. Tests pass.

### Step 5: Flatten `pending_orders` and Remove `OrderLine` Allocations
* **Action:**
  - Replace `state.pending_outbound_order_lines` and `state.pending_inbound_order_lines` with the flat compressed `pending_orders` backorder vector.
  - For tests and reporting functions that still inspect the legacy list of order lines, lazily materialize/reconstruct `OrderLine` structs on demand from the flat state, or maintain a flat primitive circular buffer inside `State` that tracks active placements without heap-allocating full objects.
* **Status:** Working to working. GC allocation overhead drops to near-zero. All correctness tests pass, but simulation runs much faster.

### Step 6: Final Cleanup and Compilation Optimization
* **Action:** Clean up any virtual dictionary wrappers and transition the last legacy files (such as visualization or optimization steps) to read from the flat data structures directly. Annotate inner loops with `@simd`, `@inbounds`, and optimize compilation.
* **Status:** Goal Achieved! The simulation code is now fully flat, cache-aligned, ready for multi-threaded scenario expansion, and capable of a 10x-100x+ execution speedup inside the same codebase.

---

## 8. Senior Developer Critique & Analysis

To ensure a highly robust, risk-mitigated implementation, a senior developer's critical evaluation of this proposal highlights several potential pitfalls, edge cases, and design improvements.

### 8.1 Type Stability and Virtual Dictionary Wrappers (Step 2 & 3 Pitfall)
* **The Pitfall:** Proposing "virtual dictionary wrappers (using a lightweight wrapper implementing `AbstractDict`)" to bridge the flat arrays with legacy code is highly elegant, but extremely prone to **type-instability** and **method dispatch overhead** in Julia. If the Julia compiler cannot infer the return type of keys or values from this virtual dict, it will fallback to dynamic dispatch and heap boxings, completely undoing the performance gains of the flat state for any bridge-using path.
* **The Fix:**
  - Refrain from writing heavy generic `AbstractDict` subtypes. Instead, utilize **type-parameterized inline accessor functions** like `get_on_hand_inventory` and `set_on_hand_inventory!` as the *only* source of truth.
  - For tests or custom policies that absolutely demand a dictionary interface, construct a dedicated `ReadOnlyDictView{K, V}` using typed parameterization, making sure all keys and values are concrete (e.g. `Tuple{Storage, Product}` and `Int64`). Ensure that methods such as `getindex` are marked with `@inline`.
  - Compile-time checking: Add `@code_warntype` assertions in tests to confirm that accessing inventory through these views does not introduce any type instabilities (`Any` types or red warning text in Julia's AST).

### 8.2 Dynamic Network Structures (Edge Case)
* **The Pitfall:** The flat layout design assumes a fixed network topology where locations, products, and lanes are immutable for the duration of a simulation or optimization run. While `simulate` operates under a fixed network, the modeling package (`SupplyChainModeling.jl`) allows calling `add_storage!`, `add_product!`, or `add_lane!` dynamically. If a user constructs a network, runs a simulation, adds a lane, and runs another simulation on the same `State` object without rebuilding the mapping, the pre-calculated integer offsets will become invalid or out-of-bounds, causing segmentation faults or silent data corruption.
* **The Fix:**
  - Implement a **topology version tracker** (a simple integer counter `topology_version` on the `SupplyChain` object) that increments whenever a lane, product, or location is added or modified.
  - `FlatEnv` and `FlatState` must cache the `topology_version` they were built for. Before executing `simulate`, verify that the cached version matches the network's current version. If there is a mismatch, raise a descriptive Julia error or trigger an automatic, fast in-place rebuild of the flat index mapping.

### 8.3 Thread Safety and Shared State in Multi-Threading (Step 6 Improvement)
* **The Pitfall:** Running scenario-level parallelism in `optimize!` via `Threads.@threads` can lead to race conditions if multiple threads modify the same `policies` parameter array or share a mutable `SimMetrics` object.
* **The Fix:**
  - Thread-local allocations: Each thread in `optimize!` must be allocated its own independent, pre-allocated `FlatState` and `FlatEnv` buffers.
  - Parameter isolation: During the optimization loop (`minimize!`), threads must read from a read-only parameter array `x` to set local parameters on their thread-specific policy copies, ensuring zero write contention on shared memory.
  - Lock-free reduction: Accumulate values across initial states / scenarios using a parallel reduction pattern rather than locking a shared accumulator variable, preserving perfect thread scaling.

### 8.4 Avoid GC Overhead in `FlatState` Allocations
* **The Pitfall:** If `FlatState` allocates fresh vectors (e.g., `Vector{Int64}(undef, M)`) inside its constructor, calling `reset!(state)` on every trial in the optimization loop will still incur substantial garbage collection overhead, as thousands of trial runs allocate and discard arrays.
* **The Fix:**
  - High-performance memory reuse: The `reset!(state)` function must **never allocate**. It should use in-place zeroing functions like `fill!(state.on_hand_totals, 0)` and `fill!(state.in_transit, 0)` on the existing pre-allocated matrices.
  - Keep the lifetime of the `FlatState` persistent across all policy evaluation runs in `bboptimize`, performing only in-place updates.
