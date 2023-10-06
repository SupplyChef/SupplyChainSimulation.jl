using PlotlyJS

function plot_inventory_onhand(state::State, location::Location, product)
    #historical_on_hand::Array{Dict{Storage, Dict{Product, Int64}}, 1}
    layout = Layout(title="Inventory on hand",
                   xaxis_title="Period",
                   yaxis_title="Unit")

    plot(1:length(state.historical_on_hand), [historical_on_hand[location][product] for historical_on_hand in state.historical_on_hand], layout)
end

function plot_inventory_onhand(states::Array{State, 1}, location::Location, product)
    #historical_on_hand::Array{Dict{Storage, Dict{Product, Int64}}, 1}
    layout = Layout(title="Inventory on hand",
                   xaxis_title="Period",
                   yaxis_title="Unit")

    plot([scatter(;x=1:length(state.historical_on_hand), 
                   y=[historical_on_hand[location][product] for historical_on_hand in state.historical_on_hand],
                   line_color=:blue,
                   opacity=0.2) for state in states], layout)
end

function plot_inventory_onhand(state::State, locations::Array{Location, 1}, product)
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

function plot_inventory_movement(state, product)
    labels = []
    sources = []
    targets = []
    values = []

    index = 0
    mapping = Dict{String, Int}()

    for i in 1:length(state.historical_filled_order_lines)
        for ol in filter(ol -> ol.product == product, state.historical_filled_order_lines[i])
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
            if get(state.historical_on_hand[i][location], product, 0) > 0
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
                push!(values, state.historical_on_hand[i][location][product])
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