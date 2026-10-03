# Scenario generation: turns one deterministic SupplyChain into many, each a
# plain SupplyChain with its own concrete data. Nothing in the model or the
# simulation knows about distributions - they are only used here, to draw
# the concrete numbers (see Lane.lead_times).

"""
    iid_lead_times(distribution)

A lead-time sampler (see [`sample_scenarios`](@ref)) that draws every departure period's lead
time independently from `distribution`, any `Distributions.jl` univariate distribution; draws
are rounded to whole periods.
"""
function iid_lead_times(distribution)
    return (rng, horizon, nominal) -> [round(Int, rand(rng, distribution)) for _ in 1:horizon]
end

"""
    constant_lead_time(distribution)

A lead-time sampler (see [`sample_scenarios`](@ref)) that draws one lead time per scenario from
`distribution` and uses it for every departure period: uncertainty about the lane's lead time,
rather than shipment-to-shipment variation.
"""
function constant_lead_time(distribution)
    return (rng, horizon, nominal) -> fill(round(Int, rand(rng, distribution)), horizon)
end

"""
    sample_scenarios(base::SupplyChain, n::Integer; seed::Integer=1, lead_times=Dict())::Vector{SupplyChain}

Builds `n` scenarios of `base`: ordinary supply chains, identical to `base` except for the
lead times of the lanes named in `lead_times`, which map a lane's `id` to a sampler. A sampler
is any function `(rng, horizon, nominal) -> Vector{Int}` returning one lead time per departure
period, where `nominal` is the lane's `times` entry; [`iid_lead_times`](@ref) and
[`constant_lead_time`](@ref) cover the common cases. Lanes into a customer cannot be named, since
nothing is shipped to customers in the simulation. For a lane with several destinations the
sampler is called once per destination.

Scenarios are reproducible: scenario `i` depends only on `seed` and `i`, so asking for more
scenarios never changes the earlier ones. A lane is the same lane in every scenario, so policies
keyed by the base supply chain's lanes apply to all of them, e.g.
`optimize!(policies, sample_scenarios(base, 100; lead_times=...)...)`.
`base` is not modified.
"""
function sample_scenarios(base::SupplyChain, n::Integer; seed::Integer=1, lead_times::AbstractDict=Dict{String, Any}())
    n >= 1 || throw(ArgumentError("n must be at least 1, got $n"))
    known_ids = Set(lane.id for lane in base.lanes if !ismissing(lane.id))
    for id in keys(lead_times)
        id in known_ids || throw(ArgumentError("no lane with id \"$id\" in the supply chain, so its lead times cannot be sampled"))
    end
    for lane in base.lanes
        if !ismissing(lane.id) && haskey(lead_times, lane.id) && any(d -> d isa Customer, lane.destinations)
            throw(ArgumentError("lane \"$(lane.id)\" delivers to a customer; lead times to customers are not simulated, so they cannot be sampled"))
        end
    end

    return map(1:n) do i
        rng = Random.Xoshiro(hash((seed, i)))
        # Drawn lane by lane in the supply chain's own lane order, never in
        # Dict/Set order, so a scenario is a pure function of (seed, i).
        sampled = Dict{String, Vector{Vector{Int}}}()
        for lane in base.lanes
            if !ismissing(lane.id) && haskey(lead_times, lane.id)
                sampler = lead_times[lane.id]
                per_destination = Vector{Int}[]
                for (d, nominal) in enumerate(lane.times)
                    draws = collect(Int, sampler(rng, base.horizon, nominal))
                    if length(draws) != base.horizon
                        throw(ArgumentError("the lead-time sampler for lane \"$(lane.id)\" returned $(length(draws)) values, expected one per period ($(base.horizon))"))
                    end
                    push!(per_destination, draws)
                end
                sampled[lane.id] = per_destination
            end
        end
        modified_copy(base; lane = lane -> !ismissing(lane.id) && haskey(sampled, lane.id) ? Lane(lane; lead_times=sampled[lane.id]) : lane)
    end
end
