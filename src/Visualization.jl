using PlotlyJS

"""
    plot_inventory_onhand(state::State, location::Location, product)

    Plots the inventory on hand of a product at a location over time.
"""
function plot_inventory_onhand(state::State, location::Node, product::Product)
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
function plot_inventory_onhand(states::Array{State, 1}, location::Node, product::Product)
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
function plot_inventory_onhand(state::State, locations::Array{L, 1}, product::Product) where L <: Node
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

# function plot_pending_outbound_order_lines(state::State, locations::Array{L, 1}, product::Product) where L <: Node
#     layout = Layout(title="Pending outbound order lines",
#                    xaxis_title="Period",
#                    yaxis_title="Unit")

#     plot([scatter(;x=1:length(state.historical_pending_outbound_order_lines), 
#                   y=[sum(ol -> (ol.order.due_date >= time) ? ol.quantity : 0, get(historical_pending_outbound_order_lines, (location, product), OrderLine[]); init=0) for (time, historical_pending_outbound_order_lines) in enumerate(state.historical_pending_outbound_order_lines)],
#                   name=location.name,
#                   mode="lines") for location in locations],
#         layout)
# end

function plot_orders(state::State, locations::Array{L, 1}, product::Product) where L <: Node
    layout = Layout(title="Orders",
                   xaxis_title="Period",
                   yaxis_title="Unit")

    plot([scatter(;x=1:get_horizon(state), 
                  y=[get_past_inbound_orders(state, location, product, t + 1, 1)[1] for t in 1:get_horizon(state)],
                  name=location.name,
                  mode="lines") for location in locations],
        layout)
end

"""
    plot_inventory_movement(state, product)

    Plots the inventory movement of a product through the supply chain through time.
"""
function plot_inventory_movement(state::State, product::Product)
    labels = []
    sources = []
    targets = []
    values = []

    index = 0
    mapping = Dict{String, Int}()

    for i in 1:length(state.historical_filled_orders)
        for ol in filter(ol -> ol.product == product, state.historical_filled_orders[i])
            source = "$(ol.order.origin.name)@$i"
            if !haskey(mapping, source)
                mapping[source] = index
                push!(labels, source)
                index = index + 1
            end

            destination = "$(ol.order.destination.name)@$(i+get_leadtime(ol.trip.route, ol.order.destination))"
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
                source = "$(location.name)@$i"
                if !haskey(mapping, source)
                    mapping[source] = index
                    push!(labels, source)
                    index = index + 1
                end

                destination = "$(location.name)@$(i+1)"
                if !haskey(mapping, destination)
                    mapping[destination] = index
                    push!(labels, destination)
                    index = index + 1
                end

                push!(sources, mapping[source])
                push!(targets, mapping[destination])
                push!(values, state.historical_on_hand[i][location][product] + 0.01)
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