"""
    TopUpRule

How a lane's total minimum order quantity (`Lane.minimum_quantity`) is reached when the
products' individually-ordered quantities add up to less than it.

The minimum is a floor on what is ordered across *all* products on the lane in a period; it
never creates an order for a product whose policy ordered nothing. Each product's own
`minimum_order_quantity` and `order_multiple` (see `SupplyChainModeling.add_product!` for
suppliers) are applied first, so a rule only ever *adds* to orders, and must add whole
multiples to keep them valid.

Pass a rule to `simulate` with `top_up_rule=...`. To define your own, subtype `TopUpRule` and
implement [`top_up!`](@ref).
"""
abstract type TopUpRule end

"""
    ProportionalTopUp()

Spreads the shortfall over the ordered products in proportion to their quantities, each share
rounded up to the product's order multiple. Keeps the mix the policies asked for; may overshoot
the minimum slightly when multiples are large. The default.
"""
struct ProportionalTopUp <: TopUpRule end

"""
    LargestOrderTopUp()

Adds the whole shortfall to the product with the largest order (the first, on ties), rounded up
to that product's order multiple. Concentrates the excess in a single product.
"""
struct LargestOrderTopUp <: TopUpRule end

"""
    top_up!(rule::TopUpRule, quantities::Vector{Int}, multiples::Vector{Int}, shortfall::Int)

Raises entries of `quantities` in place so their sum grows by at least `shortfall` (> 0).
`multiples[i]` is product `i`'s order multiple: whatever is added to `quantities[i]` must be a
multiple of it. Only entries that are already positive may be raised.
"""
function top_up!(rule::TopUpRule, quantities::Vector{Int}, multiples::Vector{Int}, shortfall::Int)
    error("top_up! is not implemented for $(typeof(rule))")
end

function top_up!(::ProportionalTopUp, quantities::Vector{Int}, multiples::Vector{Int}, shortfall::Int)
    total = sum(quantities)
    for i in eachindex(quantities)
        if quantities[i] > 0
            share = cld(shortfall * quantities[i], total)
            quantities[i] += cld(share, multiples[i]) * multiples[i]
        end
    end
    return nothing
end

function top_up!(::LargestOrderTopUp, quantities::Vector{Int}, multiples::Vector{Int}, shortfall::Int)
    i = argmax(quantities)
    quantities[i] += cld(shortfall, multiples[i]) * multiples[i]
    return nothing
end

# A product's order on a lane with a total minimum, held back by place_orders until every
# product at the location has been evaluated (see flush_pending_orders!).
struct PendingOrder
    trip::Trip
    product::Product
    pi::Int64
    quantity::Int64
    multiple::Int64
end
