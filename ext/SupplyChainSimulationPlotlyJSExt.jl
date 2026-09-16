# Package extension: loaded automatically once PlotlyJS is `using`'d
# alongside SupplyChainSimulation (see Project.toml's
# [weakdeps]/[extensions] and src/Visualization.jl's stub declarations for
# why this split exists - PlotlyJS 0.18.x needs an old JSON.jl that's
# incompatible with anything needing modern JSON.jl, e.g. Oxygen.jl).
#
# This is a straight move of what used to be src/Visualization.jl - method
# bodies are unchanged, just relocated and qualified against the parent
# module since this lives in a separate module now. `get_horizon` is the
# one call qualified with `SupplyChainSimulation.` below - it isn't
# exported by that module, so a plain `using SupplyChainSimulation`
# wouldn't bring it into scope here the way it did when this file was
# `include()`d directly into that module.
module SupplyChainSimulationPlotlyJSExt

using SupplyChainSimulation
using SupplyChainModeling
using PlotlyJS

"""
    plot_inventory_onhand(state::State, location::Location, product)

    Plots the inventory on hand of a product at a location over time.
"""
function SupplyChainSimulation.plot_inventory_onhand(state::State, location::ConcreteNode, product::Product)
    #historical_on_hand::Array{Dict{Storage, Dict{Product, Int64}}, 1}
    layout = Layout(title="Inventory on hand",
                   xaxis_title="Period",
                   yaxis_title="Unit")

    plot(1:length(state.historical_on_hand), [historical_on_hand[location][product] for historical_on_hand in state.historical_on_hand], layout)
end

"""
    plot_inventory_onhand(state::Array{State, 1}, location::Location, product)

    Plots the inventory on hand of a product at a location over time for multiple scenarios.
"""
function SupplyChainSimulation.plot_inventory_onhand(states::Array{State, 1}, location::ConcreteNode, product::Product)
    #historical_on_hand::Array{Dict{Storage, Dict{Product, Int64}}, 1}
    layout = Layout(title="Inventory on hand",
                   xaxis_title="Period",
                   yaxis_title="Unit")

    plot([scatter(;x=1:length(state.historical_on_hand),
                   y=[historical_on_hand[location][product] for historical_on_hand in state.historical_on_hand],
                   line_color=:blue,
                   opacity=0.2) for state in states], layout)
end

"""
    plot_inventory_onhand(state::State, locations::::Array{L, 1}, product) where L <: Location

    Plots the inventory on hand of a product over time for multiple locations.
"""
function SupplyChainSimulation.plot_inventory_onhand(state::State, locations::Array{L, 1}, product::Product) where L <: ConcreteNode
    #historical_on_hand::Array{Dict{Storage, Dict{Product, Int64}}, 1}
    layout = Layout(title="Inventory on hand",
                   xaxis_title="Period",
                   yaxis_title="Unit")

    plot([scatter(;x=1:length(state.historical_on_hand),
                  y=[historical_on_hand[locations[i]][product] for historical_on_hand in state.historical_on_hand],
                  name=locations[i].name,
                  mode="lines") for i in 1:length(locations)],
        layout)
end

# function plot_pending_outbound_order_lines(state::State, locations::Array{L, 1}, product::Product) where L <: ConcreteNode
#     layout = Layout(title="Pending outbound order lines",
#                    xaxis_title="Period",
#                    yaxis_title="Unit")

#     plot([scatter(;x=1:length(state.historical_pending_outbound_order_lines),
#                   y=[sum(ol -> (ol.order.due_date >= time) ? ol.quantity : 0, get(historical_pending_outbound_order_lines, (location, product), OrderLine[]); init=0) for (time, historical_pending_outbound_order_lines) in enumerate(state.historical_pending_outbound_order_lines)],
#                   name=location.name,
#                   mode="lines") for location in locations],
#         layout)
# end

function SupplyChainSimulation.plot_orders(state::State, locations::Array{L, 1}, product::Product) where L <: ConcreteNode
    layout = Layout(title="Orders",
                   xaxis_title="Period",
                   yaxis_title="Unit")

    plot([scatter(;x=1:SupplyChainSimulation.get_horizon(state),
                  y=[get_past_outbound_orders(state, location, product, t + 1, 1)[1] for t in 1:SupplyChainSimulation.get_horizon(state)],
                  name=location.name,
                  mode="lines") for location in locations],
        layout)
end

"""
    plot_inventory_movement(state, product)

    Plots the inventory movement of a product through the supply chain through time.
"""
function SupplyChainSimulation.plot_inventory_movement(state::State, product::Product)
    labels = []
    sources = []
    targets = []
    values = []

    index = 0
    mapping = Dict{String, Int}()

    for i in 1:length(state.historical_filled_orders)
        for ol in filter(ol -> ol.product == product, state.historical_filled_orders[i])
            source = "$(ol.origin.name)@$i"
            if !haskey(mapping, source)
                mapping[source] = index
                push!(labels, source)
                index = index + 1
            end

            destination = "$(ol.destination.name)@$(i+get_leadtime(ol.trip.route, ol.destination))"
            if !haskey(mapping, destination)
                mapping[destination] = index
                push!(labels, destination)
                index = index + 1
            end

            push!(sources, mapping[source])
            push!(targets, mapping[destination])
            push!(values, ol.quantity)
        end
    end

    for i in 1:length(state.historical_on_hand)-1
        for location in keys(state.historical_on_hand[i])
            if true #get(state.historical_on_hand[i][location], product, 0) > 0
                source = "$(location[1].name)@$i"
                if !haskey(mapping, source)
                    mapping[source] = index
                    push!(labels, source)
                    index = index + 1
                end

                destination = "$(location[1].name)@$(i+1)"
                if !haskey(mapping, destination)
                    mapping[destination] = index
                    push!(labels, destination)
                    index = index + 1
                end

                push!(sources, mapping[source])
                push!(targets, mapping[destination])
                push!(values, state.historical_on_hand[i][(location[1], product)] + 0.01)
            end
        end
    end

    plot(sankey(
        node = attr(
        pad = 15,
        thickness = 20,
        line = attr(color = "black", width = 0.5),
        label = labels,
        color = "blue"
        ),
        link = attr(
        source = sources, # indices correspond to labels, eg A1, A2, A1, B1, ...
        target = targets,
        value = values
    )),
    Layout(title_text="Inventory Movement", font_size=10)
    )
end

end # module
