"""
    required_lookback(policy::InventoryOrderingPolicy)::Int

The number of past periods' outbound order quantities (see
`get_past_outbound_orders`) `policy` needs to make its ordering decisions,
or `0` (the default, for every policy except `BackwardCoverageOrderingPolicy`)
if it doesn't look at history at all.

`Env` uses this at construction time to decide whether `State` needs to
maintain `outbound_order_quantities` for the run: recording it is skipped
entirely - no allocation, no per-order bookkeeping - unless some policy in
play actually declares a nonzero requirement here. Override this for any
future policy that needs to look backward the way
`BackwardCoverageOrderingPolicy` does.
"""
required_lookback(policy::InventoryOrderingPolicy)::Int = 0

"""
    safe_round_int(value::Float64)::Int64

Rounds `value` to the nearest `Int64`, or `0` if `value` is non-finite or too
large to represent as an `Int64` (`Int(round(value))` throws `InexactError`
in both cases). `set_parameters!` feeds the optimizer's raw candidate values
straight into policy fields; BlackBoxOptim's mutation/crossover can propose a
trial outside the nominal `SearchRange` (see `optimize!`), and over
thousands of evaluations one occasionally lands far enough out to overflow
here - the same class of "degenerate optimizer candidate" already guarded
against on the read side in `ForwardCoverageOrderingPolicy`/
`BackwardCoverageOrderingPolicy`'s `get_order`.
"""
function safe_round_int(value::Float64)::Int64
    if !isfinite(value) || abs(value) >= 1e15
        return 0
    end
    return round(Int64, value)
end

"""
Orders a given quantity specific to each time period.
"""
mutable struct QuantityOrderingPolicy <: InventoryOrderingPolicy
    orders::Array{Int64, 1}
end

"""
    get_parameters(policy::QuantityOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::QuantityOrderingPolicy)
    return policy.orders
end

function set_parameters!(policy::QuantityOrderingPolicy, values::Array{Float64, 1})
    policy.orders .= safe_round_int.(values)
end

function get_order(policy::QuantityOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)::Int64
    order = max(0, policy.orders[time])
    return order
end

function get_order(policy::QuantityOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, time::Int64)::Int64
    return get_order(policy, state, env, location, lane, product, 0, 0, 0, time)
end

"""
Orders a given quantity at a given time period.
"""
mutable struct ProductQuantityOrderingPolicy <: InventoryOrderingPolicy
    order::Int64
    period::Int64
end

"""
    get_parameters(policy::ProductQuantityOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::ProductQuantityOrderingPolicy)
    return [policy.order]
end

function set_parameters!(policy::ProductQuantityOrderingPolicy, values::Array{Float64, 1})
    policy.order = safe_round_int(values[1])
end

function get_order(policy::ProductQuantityOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)::Int64
    if time == policy.period
        order = max(0, policy.order)
        return order
    else
        return 0
    end 
end

function get_order(policy::ProductQuantityOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, time::Int64)::Int64
    return get_order(policy, state, env, location, lane, product, 0, 0, 0, time)
end

"""
Orders up to a given number based on the number of units on hand; no matter what is on order.
"""
mutable struct OnHandUptoOrderingPolicy <: InventoryOrderingPolicy
    upto::Int64
end


"""
    get_parameters(policy::OnHandUptoOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::OnHandUptoOrderingPolicy)
    return [policy.upto]
end

function set_parameters!(policy::OnHandUptoOrderingPolicy, values::Array{Float64, 1})
    policy.upto = safe_round_int(values[1])
end

function get_order(policy::OnHandUptoOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)::Int64
    return max(0, policy.upto - _on_hand_by_index(state, si, pi))
end

function get_order(policy::OnHandUptoOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, time::Int64)::Int64
    si = get(state.storage_index, location, 0)
    pi = get(state.product_index, product, 0)
    return get_order(policy, state, env, location, lane, product, 0, si, pi, time)
end

"""
Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog).
"""
mutable struct NetUptoOrderingPolicy <: InventoryOrderingPolicy
    upto::Int64
end


"""
    get_parameters(policy::NetUptoOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::NetUptoOrderingPolicy)
    return [policy.upto]
end

function set_parameters!(policy::NetUptoOrderingPolicy, values::Array{Float64, 1})
    policy.upto = safe_round_int(values[1])
end

function get_order(policy::NetUptoOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)::Int64
    return max(0, policy.upto - _net_inventory_by_index(state, li, pi, si, time))
end

function get_order(policy::NetUptoOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, time::Int64)::Int64
    li = get(state.location_index, location, 0)
    pi = get(state.product_index, product, 0)
    si = location isa Storage ? get(state.storage_index, location, 0) : 0
    return get_order(policy, state, env, location, lane, product, li, si, pi, time)
end

"""
Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog) if the net inventory is below a threshold.
"""
mutable struct NetSSOrderingPolicy <: InventoryOrderingPolicy
    s::Int64
    S::Int64
end

"""
    get_parameters(policy::NetSSOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::NetSSOrderingPolicy)
    return [policy.s, policy.S]
end

function set_parameters!(policy::NetSSOrderingPolicy, values::Array{Float64, 1})
    policy.s = safe_round_int(values[1])
    policy.S = safe_round_int(values[2])
end

function get_order(policy::NetSSOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)::Int64
    net_inventory = _net_inventory_by_index(state, li, pi, si, time)
    if net_inventory >= policy.s
        return 0
    else
        return max(0, policy.S - net_inventory)
    end
end

function get_order(policy::NetSSOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, time::Int64)::Int64
    li = get(state.location_index, location, 0)
    pi = get(state.product_index, product, 0)
    si = location isa Storage ? get(state.storage_index, location, 0) : 0
    return get_order(policy, state, env, location, lane, product, li, si, pi, time)
end

"""
Orders inventory to cover the coming periods based on the mean forecasted demand.
"""
mutable struct ForwardCoverageOrderingPolicy <: InventoryOrderingPolicy
    cover::Float64
end

"""
    get_parameters(policy::ForwardCoverageOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::ForwardCoverageOrderingPolicy)
    return [policy.cover]
end

function set_parameters!(policy::ForwardCoverageOrderingPolicy, values::Array{Float64, 1})
    policy.cover = values[1]
end

function get_order(policy::ForwardCoverageOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)::Int64
    net_inventory = _net_inventory_by_index(state, li, pi, si, time)
    mean_demand = get_mean_demand(env, location, product, time)

    coverage = 0
    cover = policy.cover
    t = time
    while cover > 0
        if cover >= 1
            coverage = coverage + mean_demand[min(t, end)]
            cover = cover - 1
            t = t + 1
        else
            coverage = coverage + cover * mean_demand[min(t, end)]
            cover = 0
        end
    end
    # coverage - net_inventory can come out NaN/Inf for degenerate optimizer
    # candidates (e.g. mean_demand summing to values that make later float
    # arithmetic non-finite); guard rather than let Int(ceil(...)) throw.
    deficit = coverage - net_inventory
    if !isfinite(deficit)
        @warn "ForwardCoverageOrderingPolicy.get_order: non-finite deficit, falling back to order=0" cover=policy.cover coverage net_inventory maxlog=50
    end
    # isfinite alone doesn't rule out a deficit so large that ceil(Int, ...)
    # itself throws InexactError (Int64 overflow) - a degenerate optimizer
    # candidate (e.g. cover pinned near the top of its search range across a
    # network with large aggregate demand) can reach one without ever
    # producing a NaN/Inf. 1e15 is far above any real order quantity, so this
    # doesn't change behavior for finite, in-range deficits.
    order = (isfinite(deficit) && deficit < 1e15) ? max(0, ceil(Int, deficit)) : 0
    #println("cover $(policy.cover); mean demand $mean_demand; coverage $coverage; net inventory $net_inventory; order $order; time $time")
    return order
end

function get_order(policy::ForwardCoverageOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, time::Int64)::Int64
    li = get(state.location_index, location, 0)
    pi = get(state.product_index, product, 0)
    si = location isa Storage ? get(state.storage_index, location, 0) : 0
    return get_order(policy, state, env, location, lane, product, li, si, pi, time)
end

"""
Orders inventory to cover the coming periods based on past demand.
"""
mutable struct BackwardCoverageOrderingPolicy <: InventoryOrderingPolicy
    # `Real`, not `Float64`: normal (non-AD) callers always construct/assign
    # plain `Float64` values here, so this changes nothing for them - but a
    # concretely-`Float64` field couldn't hold a `ForwardDiff.Dual` during
    # gradient-based differentiation of `optimize!`'s cost function w.r.t.
    # these parameters (see `minimize!`'s and `_backward_coverage_order`'s
    # comments for the rest of what that requires).
    cover::Vector{Real}
end

"""
    get_parameters(policy::BackwardCoverageOrderingPolicy)

    Gets the parameters for the policy.
"""
function get_parameters(policy::BackwardCoverageOrderingPolicy)
    return policy.cover
end

function set_parameters!(policy::BackwardCoverageOrderingPolicy, values::AbstractVector{<:Real})
    policy.cover = values
end

"""
    required_lookback(policy::BackwardCoverageOrderingPolicy)::Int

`get_order` below looks back `length(policy.cover)` periods via
`get_past_outbound_orders`, so that's exactly how far `Env` needs to keep
`outbound_order_quantities` for this policy's `(location, product)`.
"""
required_lookback(policy::BackwardCoverageOrderingPolicy)::Int = length(policy.cover)

# `ceil(Int, ...)`'s AbstractFloat branch is exactly today's behavior for
# every existing (non-AD) caller - production `simulate()` always calls
# `get_order` with plain `Float64` policy parameters, never anything else.
# The fallback branch only activates for other `Real` subtypes, in practice
# `ForwardDiff.Dual` during gradient-based differentiation of `optimize!`'s
# cost function: `ceil(Int, ::Dual)` (see ForwardDiff.jl's own source)
# unconditionally discards the derivative and returns a plain `Int`, which
# would zero the gradient through *every* evaluation, not just the rare
# threshold crossing where an order quantity's integer rounding is
# genuinely non-differentiable. Treating the order quantity as continuous
# during differentiation is the same simplification classical pathwise/IPA
# gradient estimators for inventory systems make (Glasserman & Tayur, 1995).
_order_quantity(x::AbstractFloat) = max(0, ceil(Int, x))
_order_quantity(x::Real) = max(zero(x), x)

# Shared by both get_order overloads below: the one difference between
# them is how lane_idx is obtained (a free field read from a Trip already
# in hand vs. a state.lane_index Dict lookup from a bare Lane) - the
# coverage math past that point is identical either way, so it lives here
# once rather than being duplicated per overload.
@inline function _backward_coverage_order(policy::BackwardCoverageOrderingPolicy, state::State, env::Env, lane_idx::Int64, li::Int64, si::Int64, pi::Int64, time::Int64)
    net_inventory = _net_inventory_by_index(state, li, pi, si, time)

    past_orders = _fill_past_outbound_orders_by_index!(env.past_orders_buffers[lane_idx, pi], state, li, pi, time)

    weights = 0.0
    coverage = 0.0
    for i in 1:length(policy.cover) - 1
        if !ismissing(past_orders[i])
            coverage += policy.cover[i] * past_orders[i]
            weights += policy.cover[i]
        end
    end

    if weights != 0.0
        coverage = coverage / (weights / sum(policy.cover))
    end

    coverage = coverage + policy.cover[end]

    # coverage - net_inventory can come out NaN/Inf for degenerate optimizer
    # candidates (e.g. weights/sum(policy.cover) dividing by ~0); guard
    # rather than let Int(ceil(...)) throw.
    deficit = coverage - net_inventory
    if !isfinite(deficit)
        @warn "BackwardCoverageOrderingPolicy.get_order: non-finite deficit, falling back to order=0" cover=policy.cover coverage weights net_inventory maxlog=50
    end
    # See the identical guard in ForwardCoverageOrderingPolicy.get_order:
    # isfinite doesn't rule out a deficit too large for ceil(Int, ...) to
    # represent without overflowing.
    return (isfinite(deficit) && deficit < 1e15) ? _order_quantity(deficit) : 0
end

"""
Fast path for `place_orders`' internal dispatch (Simulation.jl), which
already has the full `Trip` - not just its `route::Lane` - in hand: reads
`trip.lane_index` directly instead of hashing `trip.route` through
`state.lane_index` on every call. CPU profiling of the Profile workflow
found that lookup as real self-time in `simulate()`'s hot loop even after
`state.lane_index` itself was cached (see `get_lane_index`,
SupplyChainModeling.jl) - `Trip.lane_index` (Model-Transportation.jl)
resolves it once at `Env` construction instead.
"""
function get_order(policy::BackwardCoverageOrderingPolicy, state::State, env::Env, location::ConcreteNode, trip::Trip, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)
    return _backward_coverage_order(policy, state, env, trip.lane_index, li, si, pi, time)
end

"""
Lane-based fallback for direct/external callers that only have a `Lane`,
not a `Trip`, in hand (e.g. calling `get_order` outside `place_orders`).
`place_orders` itself calls the `Trip`-taking overload above instead.
"""
function get_order(policy::BackwardCoverageOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)
    return _backward_coverage_order(policy, state, env, state.lane_index[lane], li, si, pi, time)
end

function get_order(policy::BackwardCoverageOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, time::Int64)
    li = get(state.location_index, location, 0)
    pi = get(state.product_index, product, 0)
    si = location isa Storage ? get(state.storage_index, location, 0) : 0
    return get_order(policy, state, env, location, lane, product, li, si, pi, time)
end

"""
Places a single order at a given time.
"""
mutable struct SingleOrderOrderingPolicy <: InventoryOrderingPolicy
    period::Int64
    quantity
end

function get_parameters(policy::SingleOrderOrderingPolicy)
    return [policy.period, policy.quantity]
end

function set_parameters!(policy::SingleOrderOrderingPolicy, values::Array{Float64, 1})
    policy.quantity = safe_round_int(values[2])
end

function get_order(policy::SingleOrderOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)::Int64
    if time == policy.period
        return policy.quantity
    else
        return 0
    end
end

function get_order(policy::SingleOrderOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, time::Int64)::Int64
    return get_order(policy, state, env, location, lane, product, 0, 0, 0, time)
end

function get_order(policy::InventoryOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, li::Int64, si::Int64, pi::Int64, time::Int64)::Int64
    return get_order(policy, state, env, location, lane, product, time)
end

function get_order(policy::InventoryOrderingPolicy, state::State, env::Env, location::ConcreteNode, lane::Lane, product::Product, time::Int64)::Int64
    return 0
end

