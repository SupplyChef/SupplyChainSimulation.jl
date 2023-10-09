var documenterSearchIndex = {"docs":
[{"location":"#SupplyChainSimulation","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"SupplyChainSimulation is a package to model and simulate supply chains.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The simulation is built around a loop that keeps track of orders and inventory movements:","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"receive_inventory\nplace_orders\nreceive_orders\nsend_inventory","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Each step can be customized by applying different policies. The policies can also be optimized to improve the supply chain performance.","category":"page"},{"location":"#Installation","page":"SupplyChainSimulation","title":"Installation","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"SupplyChainSimulation can be installed using the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"pkg> add SupplyChainSimulation","category":"page"},{"location":"#Getting-Started","page":"SupplyChainSimulation","title":"Getting Started","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The first step to use SupplyChainSimulation is to define the supply chain. This is done by specifying the supply chain network, including product, suppliers, storage locations, and customers.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"In the example below we define a network with one product, one supplier, one storage location, and one customer.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"horizon = 20\n  \nproduct = Single(\"product\")\n\nsupplier = Supplier(\"supplier\")\nstorage = Storage(\"storage\", Dict(product => 1.0))\ncustomer = Customer(\"customer\")\n\nl1 = Lane(; origin = storage, destination = customer)\nl2 = Lane(; origin = supplier, destination = storage)\n\nnetwork = Network([supplier], [storage], [customer], get_trips([l1, l2], horizon), [product])","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The second step is to define the starting state. The initial state represents the inventory situation at the start of the simulation: what inventory is on hand, what inventory is in transit. The state also has information about the future: what demand we expect and what policies we want to use when computing orders. More than one initial state can be defined to represent potential different situations that have to be simulated or optimized. For example we could have several demand scenarios. In our example, we will create 10 such scenarios with different demand from the customer. In the example we use an order up to policy that will place an order to replenish the inventory back to a given value.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"policy = OnHandUptoOrderingPolicy(0)\n\ninitial_states = [State(; on_hand_inventory = Dict(storage => Dict(product => 0)), \n                        demand = Dict((customer, product) => rand(Poisson(10), horizon)),\n                        policies = Dict((l2, product) => policy)) for i in 1:10]","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The third step is to run the simulation or the optimization (depending on whether you already know the policies you want to use or whether you want to find the best policies). In our example we will search the best policy by running the optimizer.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"optimize!(network, horizon, initial_states...)\nfinal_states = [simulate(network, horizon, initial_state) for initial_state in initial_states]\n","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The final step is to analyze the results. There are various function we can call to get information such as the orders that have been placed, the inventory on hand at any time, and more. There are also plotting functions which provide the information in a graphical way. In our example we will plot the amount of inventory at the storage location over time.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"plot_inventory_onhand(final_states, storage, product)","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The resulting plot is shown below.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"(Image: example inventory on hand)","category":"page"},{"location":"#Policies","page":"SupplyChainSimulation","title":"Policies","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The package comes with several policies predefined, including:","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"QuantityOrderingPolicy: Orders a given quantity specific to each time period. The quantity ordered is the same across scenarios and irrespective of the current inventory position.\nOnHandUptoOrderingPolicy: Orders up to a given number based on the number of units on hand; no matter what is on order.\nNetUptoOrderingPolicy: Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog).\nNetSSOrderingPolicy: Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog) if the net inventory is below a threshold.","category":"page"},{"location":"#Creating-a-new-policy","page":"SupplyChainSimulation","title":"Creating a new policy","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Creating a new policy is easy and can be done in two steps:","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Create a new struct to represent your policy. This struct can hold the data you need to compute the orders. For example, let's say that we want to create a policy that orders always the same amount. We will create the following strut","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"mutable struct SameOrderPolicy <: InventoryOrderingPolicy\n    order::Int64\nend","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Implement the following functions:","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"function get_parameter_count(policy::SameOrderPolicy)\n    return 1 # indicates how many parameters the policy has so that the optimizer can optimize the policy.\nend\n\nfunction set_parameter!(policy::SameOrderPolicy, values::Array{Float64, 1})\n    policy.order = Int.(round.(values[1])) # sets the value of the order to the value provided by the optimizer\nend\n\nfunction get_order(policy::SameOrderPolicy, state, env, location, lane, product, time)\n    return policy.order # returns the order when running the policy during a simulation\nend","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Once the policy is defined it can be used either as part of a simulation run where the order is defined by you, or as part of an optimization run where the optimizer will find the best order to minimize the cost function.","category":"page"},{"location":"#API","page":"SupplyChainSimulation","title":"API","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Modules = [SupplyChainSimulation]\nOrder   = [:type, :function]","category":"page"},{"location":"#SupplyChainSimulation.Env","page":"SupplyChainSimulation","title":"SupplyChainSimulation.Env","text":"Contains information about the environment of the simulation, including the network configuration.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.ForwardCoverageOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.ForwardCoverageOrderingPolicy","text":"Orders inventory to cover the coming periods based on the mean forecasted demand.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.NetSSOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.NetSSOrderingPolicy","text":"Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog) if the net inventory is below a threshold.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.NetUptoOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.NetUptoOrderingPolicy","text":"Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog).\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.OnHandUptoOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.OnHandUptoOrderingPolicy","text":"Orders up to a given number based on the number of units on hand; no matter what is on order.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.QuantityOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.QuantityOrderingPolicy","text":"Orders a given quantity specific to each time period.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.State","page":"SupplyChainSimulation","title":"SupplyChainSimulation.State","text":"Contains information about the current state of the simulation, including inventory positions and pending orders.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.Trip","page":"SupplyChainSimulation","title":"SupplyChainSimulation.Trip","text":"A trip is the basis of transportation in the simulation. It follows a route with a given departure time.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.eoq_cost_rate-Tuple{Any, Any, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.eoq_cost_rate","text":"eoq_cost_rate(demand_rate, ordering_cost, holding_cost_rate)\n\nComputes the total cost per time period of ordering the economic ordering quantity.\n\nSee also [`eoq_quantity`](@ref).\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.eoq_interval-Tuple{Any, Any, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.eoq_interval","text":"eoq_interval(demand_rate, ordering_cost, holding_cost_rate)\n\nComputes at what interval the economic ordering quantity is ordered.\n\nSee also [`eoq_quantity`](@ref).\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.eoq_quantity-NTuple{4, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.eoq_quantity","text":"eoq_quantity(demand_rate, ordering_cost, holding_cost_rate, backlog_cost_rate)\n\nComputes the economic ordering quantity that minimizes overall costs (ordering costs + holding costs) while meeting  demand.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.eoq_quantity-Tuple{Any, Any, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.eoq_quantity","text":"eoq_quantity(demand_rate, ordering_cost, holding_cost_rate)\n\nComputes the economic ordering quantity that minimizes overall costs (ordering costs + holding costs) while meeting  demand.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_locations-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_locations","text":"get_locations(network)\n\nGets all the locations in the network.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameter_count-Tuple{ForwardCoverageOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameter_count","text":"get_parameter_count(policy::ForwardCoverageOrderingPolicy)\n\nGets the number of parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameter_count-Tuple{NetSSOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameter_count","text":"get_parameter_count(policy::NetSSOrderingPolicy)\n\nGets the number of parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameter_count-Tuple{NetUptoOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameter_count","text":"get_parameter_count(policy::NetUptoOrderingPolicy)\n\nGets the number of parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameter_count-Tuple{OnHandUptoOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameter_count","text":"get_parameter_count(policy::OnHandUptoOrderingPolicy)\n\nGets the number of parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameter_count-Tuple{QuantityOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameter_count","text":"get_parameter_count(policy::QuantityOrderingPolicy)\n\nGets the number of parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_demand-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_demand","text":"get_total_demand(state)\n\nGets the total demand (in unit) for the state.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_holding_costs-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_holding_costs","text":"get_total_holding_costs(state)\n\nGets the total holding costs for the state.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_lost_sales-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_lost_sales","text":"get_total_lost_sales(state)\n\nGets the total lost sales (in unit) for the state.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_sales-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_sales","text":"get_total_sales(state)\n\nGets the total sales (in unit) for the state.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.optimize!-Tuple{Network, Int64, Vararg{Any}}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.optimize!","text":"optimize!(network::Network, horizon::Int64, initial_states...; cost_function)\n\nOptimizes the inventory policies in the network by simulating the inventory movement starting from the initial states and costing the results with the cost function.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.receive_orders!-Tuple{State, SupplyChainSimulation.Env, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.receive_orders!","text":"receive_orders!(state::State, env::Env, orders)\n\nReceives the orders that have been placed.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.simulate-Tuple{SupplyChainSimulation.Env, Int64, State}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.simulate","text":"simulate(env::Env, horizon::Int64, initial_state::State)\n\nSimulates the supply chain for horizon steps, starting from the initial state.\n\n\n\n\n\n","category":"method"}]
}
