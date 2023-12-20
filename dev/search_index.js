var documenterSearchIndex = {"docs":
[{"location":"#SupplyChainSimulation","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"SupplyChainSimulation is a package to model and simulate supply chains.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The simulation is built around a loop that keeps track of orders and inventory movements:","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"receive_inventory\nplace_orders\nreceive_orders\nsend_inventory","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Each step can be customized by applying different policies. The policies can also be optimized to improve the supply chain performance.","category":"page"},{"location":"#Installation","page":"SupplyChainSimulation","title":"Installation","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"SupplyChainSimulation can be installed using the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"pkg> add SupplyChainSimulation","category":"page"},{"location":"#Getting-started","page":"SupplyChainSimulation","title":"Getting started","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The first step to use SupplyChainSimulation is to define the supply chain. This is done by specifying the supply chain network, including product, suppliers, storage locations, and customers.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"In the example below we define a network with one product, one supplier, one storage location, and one customer.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"horizon = 20\n  \nproduct = Product(\"product\")\n\nsupplier = Supplier(\"supplier\")\nstorage = Storage(\"storage\")\nadd_product!(storage, product; unit_holding_cost=1.0)\ncustomer = Customer(\"customer\")\n\nl1 = Lane(storage, customer)\nl2 = Lane(supplier, storage)\n\nnetwork = SupplyChain(horizon)\nadd_supplier!(network, supplier)\nadd_storage!(network, storage)\nadd_customer!(network, customer)\nadd_product!(network, product)\nadd_lane!(network, l1)\nadd_lane!(network, l2)","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The second step is to define the starting state. The initial state represents the inventory situation at the start of the simulation. The state also has information about the future: what demand we expect and what policies we want to use when computing orders. More than one initial state can be defined to represent potential different situations that have to be simulated or optimized. For example we could have several demand scenarios. In our example, we will create 10 such scenarios with different demand from the customer. In the example we use an order up to policy that will place an order to replenish the inventory back to a given value.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"policy = OnHandUptoOrderingPolicy(0)\n    policies = Dict((l2, product) => policy)\n\n    initial_states = [State(; demand = [Demand(customer, product, rand(Poisson(10), horizon) * 1.0; sales_price=1.0, lost_sales_cost=1.0)]) for i in 1:10]","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The third step is to run the simulation or the optimization (depending on whether you already know the policies you want to use or whether you want to find the best policies). In our example we will search the best policy by running the optimizer.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"optimize!(network, policies, initial_states...)\nfinal_states = [simulate(network, policies, initial_state) for initial_state in initial_states]","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The final step is to analyze the results. There are various function we can call to get information such as the orders that have been placed, the inventory on hand at any time, and more. There are also plotting functions which provide the information in a graphical way. In our example we will plot the amount of inventory at the storage location over time.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"plot_inventory_onhand(final_states, storage, product)","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The resulting plot is shown below.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"(Image: example inventory on hand)","category":"page"},{"location":"#Policies","page":"SupplyChainSimulation","title":"Policies","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The package comes with several policies predefined, including:","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"QuantityOrderingPolicy: Orders a given quantity specific to each time period. The quantity ordered is the same across scenarios and irrespective of the current inventory position.\nOnHandUptoOrderingPolicy: Orders up to a given number based on the number of units on hand; no matter what is on order.\nNetUptoOrderingPolicy: Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog).\nNetSSOrderingPolicy: Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog) if the net inventory is below a threshold.","category":"page"},{"location":"#Creating-a-new-policy","page":"SupplyChainSimulation","title":"Creating a new policy","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Creating a new policy is easy and can be done in two steps:","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Create a new struct to represent your policy. This struct can hold the data you need to compute the orders. For example, let's say that we want to create a policy that orders always the same amount. We will create the following strut","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"mutable struct SameOrderPolicy <: InventoryOrderingPolicy\n    order::Int64\nend","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Implement the following functions:","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"function get_parameters(policy::SameOrderPolicy)\n    return [policy.order]\nend\n\nfunction set_parameters!(policy::SameOrderPolicy, values::Array{Float64, 1})\n    policy.order = Int.(round.(values[1])) # sets the value of the order to the value provided by the optimizer\nend\n\nfunction get_order(policy::SameOrderPolicy, state, env, location, lane, product, time)\n    return policy.order # returns the order when running the policy during a simulation\nend","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Once the policy is defined it can be used either as part of a simulation run where the order is defined by you, or as part of an optimization run where the optimizer will find the best order to minimize the cost function.","category":"page"},{"location":"#Common-models","page":"SupplyChainSimulation","title":"Common models","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"In this section we will review some common inventory models and how to implement them with SupplyChainSimulation.jl.","category":"page"},{"location":"#Economic-Ordering-Quantity-(EOQ)","page":"SupplyChainSimulation","title":"Economic Ordering Quantity (EOQ)","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The Economic Ordering Quantity helps us balance the cost of ordering and the cost of holding inventory by optimizing the order quantity. The higher the ordering cost, the less often we want to order and therefore the bigger the quantity we want to order. The higher the holding cost, the less inventory we want to order and therefore the smaller the quantity we want to order. The EOQ is the order quantity that best balances these two costs. In the example below, we will consider a demand of 10 units per period, an ordering cost of 10 and a holding cost of .1 per unit per period.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"We can get the EOQ by running the following code.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"eoq_quantity(10, 10, 0.1)","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The answer is approximately 44.7.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"We can also use the simulation as follows.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"horizon = 50\n\nproduct = Product(\"product\")\n\nsupplier = Supplier(\"supplier\")\nstorage = Storage(\"storage\")\nadd_product!(storage, product; unit_holding_cost=0.1)\n\ncustomer = Customer(\"customer\")\n\nl1 = Lane(storage, customer)\nl2 = Lane(supplier, storage, fixed_cost=10)\n\nnetwork = SupplyChain(horizon)\nadd_supplier!(network, supplier)\nadd_storage!(network, storage)\nadd_customer!(network, customer)\nadd_product!(network, product)\nadd_lane!(network, l1)\nadd_lane!(network, l2)\n\npolicy = NetSSOrderingPolicy(0, 0)\npolicies = Dict((l2, product) => policy)\n\ninitial_states = [State(; demand = [Demand(customer, product, repeat([10.0], horizon); sales_price=1.0, lost_sales_cost=1.0)]) for i in 1:1]\n\noptimize!(network, policies, initial_states...)\n\nprintln(policy)","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The result is a policy (0, 40). We order 40 units when the inventory goes down to 0. This matches the EOQ we computed above.","category":"page"},{"location":"#Safety-Stock","page":"SupplyChainSimulation","title":"Safety Stock","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Let's extend the example above by making the demand stochastic and have a lead time of 2 period for the storage replenishment. The code is very similar to that above and looks as follows.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"horizon = 50\n\nproduct = Product(\"product\")\n\nsupplier = Supplier(\"supplier\")\nstorage = Storage(\"storage\")\nadd_product!(storage, product; unit_holding_cost=0.1)\ncustomer = Customer(\"customer\")\n\nl1 = Lane(storage, customer)\nl2 = Lane(supplier, storage; fixed_cost=10, time=2)\n\nnetwork = SupplyChain(horizon)\nadd_supplier!(network, supplier)\nadd_storage!(network, storage)\nadd_customer!(network, customer)\nadd_product!(network, product)\nadd_lane!(network, l1)\nadd_lane!(network, l2)\n\npolicy = NetSSOrderingPolicy(0, 0)\npolicies = Dict((l2, product) => policy)\n\ninitial_states = [State(; demand = [Demand(customer, product, rand(Poisson(10), horizon) * 1.0; ; sales_price=1.0, lost_sales_cost=1.0)]) for i in 1:20]\n\noptimize!(network, policies, initial_states...)\n\nprintln(policy)","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Now the best policy is (20, 60). The safety stock of 20 helps avoid stock out while the reorder quantity stays 40 units above the safety stock as in the EOQ example above.","category":"page"},{"location":"#Beer-game","page":"SupplyChainSimulation","title":"Beer game","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The beer game is a common supply chain setup used to teach inventory management. The supply chain is composed of 5 entities: a customer, a retailer, a wholesaler, a factory and a supplier. There is a lead time between each echelon in the supply chain. The question is how best to manage this supply chain.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"We can mode this setup with SupplyChainSimulation.jl as follows.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"product = Product(\"product\")\n\ncustomer = Customer(\"customer\")\nretailer = Storage(\"retailer\")\nadd_product!(retailer, product; initial_inventory=20, unit_holding_cost=0.1)\nwholesaler = Storage(\"wholesaler\")\nadd_product!(wholesaler, product; initial_inventory=20, unit_holding_cost=0.1)\nfactory = Storage(\"factory\")\nadd_product!(factory, product; initial_inventory=20, unit_holding_cost=0.1)\nsupplier = Supplier(\"supplier\")\n\nhorizon = 20\n\nl = Lane(retailer, customer; unit_cost=0)\nl2 = Lane(wholesaler, retailer; unit_cost=0, time=2)\nl3 = Lane(factory, wholesaler; unit_cost=0, time=2)\nl4 = Lane(supplier, factory; unit_cost=0, time=4)\n\npolicy2 = NetUptoOrderingPolicy(0)\npolicy3 = NetUptoOrderingPolicy(0)\npolicy4 = NetUptoOrderingPolicy(0)\npolicies = Dict(\n                (l2, product) => policy2,\n                (l3, product) => policy3,\n                (l4, product) => policy4)\n\nnetwork = SupplyChain(horizon)\nadd_supplier!(network, supplier)\nadd_storage!(network, factory)\nadd_storage!(network, wholesaler)\nadd_storage!(network, retailer)\nadd_customer!(network, customer)\nadd_product!(network, product)\nadd_lane!(network, l)\nadd_lane!(network, l2)\nadd_lane!(network, l3)\nadd_lane!(network, l4)\n\ninitial_states = [State(; demand = [Demand(customer, product, rand(Poisson(10), horizon) * 1.0; ; sales_price=1.0, lost_sales_cost=1.0)]) for i in 1:30]\n\noptimize!(network, policies, initial_states...)","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"The optimizer will then run and return the best policies.","category":"page"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"If you run this code you will see that the policies do extremely well with no bullwhip effect. Inventory management solved? Not fully. Let's note that (1) the policies are tuned to a specific scenario (albeit stochastic) and (2) the optimizer optimizes across echelons (as if the whole supply chain is integrated). This is a best case scenario. Different setups can be tested. For example you can add more scenarios or you can change the policies to limit what they can see. Depending on the setup the bullwhip effect can be more or less strong. Being able to simulate these results is on key advantage of using SupplyChainSimulation.jl.","category":"page"},{"location":"#API","page":"SupplyChainSimulation","title":"API","text":"","category":"section"},{"location":"","page":"SupplyChainSimulation","title":"SupplyChainSimulation","text":"Modules = [SupplyChainSimulation]\nOrder   = [:type, :function]","category":"page"},{"location":"#SupplyChainSimulation.BackwardCoverageOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.BackwardCoverageOrderingPolicy","text":"Orders inventory to cover the coming periods based on past demand.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.Env","page":"SupplyChainSimulation","title":"SupplyChainSimulation.Env","text":"Contains information about the environment of the simulation, including the network configuration.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.ForwardCoverageOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.ForwardCoverageOrderingPolicy","text":"Orders inventory to cover the coming periods based on the mean forecasted demand.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.NetSSOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.NetSSOrderingPolicy","text":"Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog) if the net inventory is below a threshold.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.NetUptoOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.NetUptoOrderingPolicy","text":"Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog).\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.OnHandUptoOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.OnHandUptoOrderingPolicy","text":"Orders up to a given number based on the number of units on hand; no matter what is on order.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.ProductQuantityOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.ProductQuantityOrderingPolicy","text":"Orders a given quantity at a given time period.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.QuantityOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.QuantityOrderingPolicy","text":"Orders a given quantity specific to each time period.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.SingleOrderOrderingPolicy","page":"SupplyChainSimulation","title":"SupplyChainSimulation.SingleOrderOrderingPolicy","text":"Places a single order at a given time.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.State","page":"SupplyChainSimulation","title":"SupplyChainSimulation.State","text":"Contains information about the current state of the simulation, including inventory positions and pending orders.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.Trip","page":"SupplyChainSimulation","title":"SupplyChainSimulation.Trip","text":"A trip is the basis of transportation in the simulation. It follows a route with a given departure time.\n\n\n\n\n\n","category":"type"},{"location":"#SupplyChainSimulation.eoq_cost_rate-Tuple{Any, Any, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.eoq_cost_rate","text":"eoq_cost_rate(demand_rate, ordering_cost, holding_cost_rate)\n\nComputes the total cost per time period of ordering the economic ordering quantity.\n\nSee also [`eoq_quantity`](@ref).\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.eoq_interval-Tuple{Any, Any, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.eoq_interval","text":"eoq_interval(demand_rate, ordering_cost, holding_cost_rate)\n\nComputes at what interval the economic ordering quantity is ordered.\n\nSee also [`eoq_quantity`](@ref).\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.eoq_quantity-NTuple{4, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.eoq_quantity","text":"eoq_quantity(demand_rate, ordering_cost, holding_cost_rate, backlog_cost_rate)\n\nComputes the economic ordering quantity that minimizes overall costs (ordering costs + holding costs) while meeting  demand.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.eoq_quantity-Tuple{Any, Any, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.eoq_quantity","text":"eoq_quantity(demand_rate, ordering_cost, holding_cost_rate)\n\nComputes the economic ordering quantity that minimizes overall costs (ordering costs + holding costs) while meeting  demand.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_horizon-Tuple{State}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_horizon","text":"get_horizon(state::State)\n\nGets the number of steps in the simulation.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_in_transit_inventory-Union{Tuple{N}, Tuple{State, N, Product, Int64}} where N<:SupplyChainModeling.Node","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_in_transit_inventory","text":"get_in_transit_inventory(state::State, to::Location, product::Product, time::Int64)::Int64\n\nGets the number of units of a product in transit to a location at a given time.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_inbound_orders-Tuple{State, SupplyChainModeling.Node, Product, Int64}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_inbound_orders","text":"get_inbound_orders(state::State, location::Location, product::Product, time::Int64)::Int64\n\nGets the number of units of a product on order to a location (but not yet shipped there) at a given time.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_locations-Tuple{SupplyChainModeling.SupplyChain}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_locations","text":"get_locations(supplychain)\n\nGets all the locations in the supplychain.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_outbound_orders-Tuple{State, SupplyChainModeling.Node, Product, Int64}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_outbound_orders","text":"get_outbound_orders(state::State, location::Location, product::Product, time::Int64)::Int64\n\nGets the number of units of a product on order at a location (and not yet shipped out) at a given time.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameters-Tuple{BackwardCoverageOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameters","text":"get_parameters(policy::BackwardCoverageOrderingPolicy)\n\nGets the parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameters-Tuple{ForwardCoverageOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameters","text":"get_parameters(policy::ForwardCoverageOrderingPolicy)\n\nGets the parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameters-Tuple{NetSSOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameters","text":"get_parameters(policy::NetSSOrderingPolicy)\n\nGets the parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameters-Tuple{NetUptoOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameters","text":"get_parameters(policy::NetUptoOrderingPolicy)\n\nGets the parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameters-Tuple{OnHandUptoOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameters","text":"get_parameters(policy::OnHandUptoOrderingPolicy)\n\nGets the parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameters-Tuple{QuantityOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameters","text":"get_parameters(policy::QuantityOrderingPolicy)\n\nGets the parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_parameters-Tuple{SupplyChainSimulation.ProductQuantityOrderingPolicy}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_parameters","text":"get_parameters(policy::ProductQuantityOrderingPolicy)\n\nGets the parameters for the policy.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_demand-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_demand","text":"get_total_demand(state)\n\nGets the total demand (in unit) for the state.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_holding_costs-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_holding_costs","text":"get_total_holding_costs(state)\n\nGets the total holding costs for the state.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_lost_sales-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_lost_sales","text":"get_total_lost_sales(state)\n\nGets the total lost sales (in unit) for the state.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_sales-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_sales","text":"get_total_sales(state)\n\nGets the total sales (in unit) for the state.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_trip_fixed_costs-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_trip_fixed_costs","text":"get_total_trip_fixed_costs(state)\n\nGets the total transportation fixed costs.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.get_total_trip_unit_costs-Tuple{Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.get_total_trip_unit_costs","text":"get_total_trip_unit_costs(state)\n\nGets the total transportation unit costs.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.optimize!-Tuple{SupplyChainModeling.SupplyChain, Dict{Tuple{SupplyChainModeling.Lane, Product}, <:InventoryOrderingPolicy}, Vararg{Any}}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.optimize!","text":"optimize!(supplychain::SupplyChain, lane_policies, initial_states...; cost_function)\n\nOptimizes the inventory policies in the supply chain by simulating the inventory movement starting from the initial states and costing the results with the cost function.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.plot_inventory_movement-Tuple{State, Product}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.plot_inventory_movement","text":"plot_inventory_movement(state, product)\n\nPlots the inventory movement of a product through the supply chain through time.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.plot_inventory_onhand-Tuple{State, SupplyChainModeling.Node, Product}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.plot_inventory_onhand","text":"plot_inventory_onhand(state::State, location::Location, product)\n\nPlots the inventory on hand of a product at a location over time.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.plot_inventory_onhand-Tuple{Vector{State}, SupplyChainModeling.Node, Product}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.plot_inventory_onhand","text":"plot_inventory_onhand(state::Array{State, 1}, location::Location, product)\n\nPlots the inventory on hand of a product at a location over time for multiple scenarios.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.plot_inventory_onhand-Union{Tuple{L}, Tuple{State, Vector{L}, Product}} where L<:SupplyChainModeling.Node","page":"SupplyChainSimulation","title":"SupplyChainSimulation.plot_inventory_onhand","text":"plot_inventory_onhand(state::State, locations::::Array{L, 1}, product) where L <: Location\n\nPlots the inventory on hand of a product over time for multiple locations.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.receive_orders!-Tuple{State, SupplyChainSimulation.Env, Any}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.receive_orders!","text":"receive_orders!(state::State, env::Env, orders)\n\nReceives the orders that have been placed.\n\n\n\n\n\n","category":"method"},{"location":"#SupplyChainSimulation.simulate-Tuple{SupplyChainSimulation.Env, Any, State}","page":"SupplyChainSimulation","title":"SupplyChainSimulation.simulate","text":"simulate(env::Env, policies, initial_state::State)\n\nSimulates the supply chain for horizon steps, starting from the initial state.\n\n\n\n\n\n","category":"method"}]
}
