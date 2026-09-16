# PlotlyJS is a weak dependency (see Project.toml's [weakdeps]/
# [extensions]): pulling it in unconditionally made this package
# unresolvable alongside anything depending on a modern JSON.jl (PlotlyJS
# 0.18.x needs JSON 0.20-0.21; e.g. Oxygen.jl needs JSON 1.x - no version
# satisfies both, so any consumer needing both would fail to resolve at
# all, not just at plot time). Same fix, same reason, as
# SupplyChainOptimization.jl's own PlotlyJS/Plots extension split.
#
# The actual implementations live in ext/SupplyChainSimulationPlotlyJSExt.jl,
# which Julia loads automatically once PlotlyJS is `using`'d alongside this
# package - no special syntax needed by the caller beyond having it loaded.
# These are just stub declarations: they keep the names part of this
# package's own export/API surface and documented, regardless of whether
# the extension is active. Calling one before PlotlyJS is loaded raises a
# plain `MethodError`.

"""
    plot_inventory_onhand(state::State, location::Location, product)

    Plots the inventory on hand of a product at a location over time.

Requires `PlotlyJS` to be loaded (see this file's top-of-file note).
"""
function plot_inventory_onhand end

"""
    plot_orders(state::State, locations::Array{L, 1}, product) where L <: ConcreteNode

    Plots outbound orders for a product across multiple locations over time.

Requires `PlotlyJS` to be loaded (see this file's top-of-file note).
"""
function plot_orders end

"""
    plot_inventory_movement(state, product)

    Plots the inventory movement of a product through the supply chain through time.

Requires `PlotlyJS` to be loaded (see this file's top-of-file note).
"""
function plot_inventory_movement end
