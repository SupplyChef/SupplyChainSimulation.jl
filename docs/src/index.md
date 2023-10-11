# SupplyChainSimulation

SupplyChainSimulation is a package to model and simulate supply chains.

The simulation is built around a loop that keeps track of orders and inventory movements:

- receive_inventory
- place_orders
- receive_orders
- send_inventory

Each step can be customized by applying different policies. The policies can also be optimized to improve the supply chain performance.

## Installation

SupplyChainSimulation can be installed using the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add SupplyChainSimulation
```

## Getting started

The first step to use SupplyChainSimulation is to define the supply chain. This is done by specifying the supply chain network, including product, suppliers, storage locations, and customers.

In the example below we define a network with one product, one supplier, one storage location, and one customer.

```julia
horizon = 20
  
product = Single("product")

supplier = Supplier("supplier")
storage = Storage("storage", Dict(product => 1.0))
customer = Customer("customer")

l1 = Lane(; origin = storage, destination = customer)
l2 = Lane(; origin = supplier, destination = storage)

network = Network([supplier], [storage], [customer], get_trips([l1, l2], horizon), [product])
```

The second step is to define the starting state. The initial state represents the inventory situation at the start of the simulation: what inventory is on hand, what inventory is in transit. The state also has information about the future: what demand we expect and what policies we want to use when computing orders. More than one initial state can be defined to represent potential different situations that have to be simulated or optimized. For example we could have several demand scenarios. In our example, we will create 10 such scenarios with different demand from the customer. In the example we use an order up to policy that will place an order to replenish the inventory back to a given value.

```julia
policy = OnHandUptoOrderingPolicy(0)

initial_states = [State(; on_hand_inventory = Dict(storage => Dict(product => 0)), 
                        demand = Dict((customer, product) => rand(Poisson(10), horizon)),
                        policies = Dict((l2, product) => policy)) for i in 1:10]
```

The third step is to run the simulation or the optimization (depending on whether you already know the policies you want to use or whether you want to find the best policies). In our example we will search the best policy by running the optimizer.

```julia
optimize!(network, horizon, initial_states...)
final_states = [simulate(network, horizon, initial_state) for initial_state in initial_states]

```

The final step is to analyze the results. There are various function we can call to get information such as the orders that have been placed, the inventory on hand at any time, and more. There are also plotting functions which provide the information in a graphical way. In our example we will plot the amount of inventory at the storage location over time.

```julia
plot_inventory_onhand(final_states, storage, product)
```

The resulting plot is shown below.

![example inventory on hand](example_inventory_on_hand.png)

## Policies

The package comes with several policies predefined, including:

- QuantityOrderingPolicy: Orders a given quantity specific to each time period. The quantity ordered is the same across scenarios and irrespective of the current inventory position.
- OnHandUptoOrderingPolicy: Orders up to a given number based on the number of units on hand; no matter what is on order.
- NetUptoOrderingPolicy: Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog).
- NetSSOrderingPolicy: Orders up to a given number based on the net number of units (on hand + in transit + on order - on backlog) if the net inventory is below a threshold.

## Creating a new policy

Creating a new policy is easy and can be done in two steps:

1. Create a new struct to represent your policy. This struct can hold the data you need to compute the orders. For example, let's say that we want to create a policy that orders always the same amount. We will create the following strut

```julia
mutable struct SameOrderPolicy <: InventoryOrderingPolicy
    order::Int64
end
```

2. Implement the following functions:

```julia
function get_parameter_count(policy::SameOrderPolicy)
    return 1 # indicates how many parameters the policy has so that the optimizer can optimize the policy.
end

function set_parameter!(policy::SameOrderPolicy, values::Array{Float64, 1})
    policy.order = Int.(round.(values[1])) # sets the value of the order to the value provided by the optimizer
end

function get_order(policy::SameOrderPolicy, state, env, location, lane, product, time)
    return policy.order # returns the order when running the policy during a simulation
end
```

Once the policy is defined it can be used either as part of a simulation run where the order is defined by you, or as part of an optimization run where the optimizer will find the best order to minimize the cost function.

## Common models

In this section we will review some common inventory models and how to implement them with SupplyChainSimulation.jl.

### Economic Ordering Quantity (EOQ)
The Economic Ordering Quantity helps us balance the cost of ordering and the cost of holding inventory by optimizing the order quantity. The higher the ordering cost, the less often we want to order and therefore the bigger the quantity we want to order. The higher the holding cost, the less inventory we want to order and therefore the smaller the quantity we want to order. The EOQ is the order quantity that best balances these two costs. In the example below, we will consider a demand of 10 units per period, an ordering cost of 10 and a holding cost of .1 per unit per period.

We can get the EOQ by running the following code.
```julia
eoq_quantity(10, 10, 0.1)
```

The answer is approximately 44.7.

We can also use the simulation as follows.

```julia
horizon = 50

product = Single("product")

supplier = Supplier("supplier")
storage = Storage("storage", Dict(product => 0.1))

customer = Customer("customer")
  
l1 = Lane(; origin = storage, destination = customer)
l2 = Lane(; origin = supplier, destination = storage, fixed_cost=10)

network = Network([supplier], [storage], [customer], get_trips([l1, l2], horizon), [product])

policy = NetSSOrderingPolicy(0, 0)

initial_states = [State(; on_hand_inventory = Dict(storage => Dict(product => 0)), 
                        demand = Dict((customer, product) => repeat([10], horizon)),
                        policies = Dict((l2, product) => policy)) for i in 1:1]

optimize!(network, horizon, initial_states...)

println(policy)
```

The result is a policy (0, 40). We order 40 units when the inventory goes down to 0. This matches the EOQ we computed above.

### Safety Stock
Let's extend the example above by making the demand stochastic and have a lead time of 2 period for the storage replenishment. The code is very similar to that above and looks as follows.

```julia
horizon = 50

product = Single("product")

supplier = Supplier("supplier")
storage = Storage("storage", Dict(product => 0.1))
customer = Customer("customer")

l1 = Lane(; origin=storage, destination=customer)
l2 = Lane(; origin=supplier, destination=storage, fixed_cost=10, lead_time=2)

network = Network([supplier], [storage], [customer], get_trips([l1, l2], horizon), [product])

policy = NetSSOrderingPolicy(0, 0)

initial_states = [State(; on_hand_inventory = Dict(storage => Dict(product => 0)), 
                        demand = Dict((customer, product) => rand(Poisson(10), horizon)),
                        policies = Dict((l2, product) => policy)) for i in 1:20]

optimize!(network, horizon, initial_states...)

println(policy)
```
Now the best policy is (20, 60). The safety stock of 20 helps avoid stock out while the reorder quantity stays 40 units above the safety stock as in the EOQ example above.

### Beer game
The beer game is a common supply chain setup used to teach inventory management. The supply chain is composed of 5 entities: a customer, a retailer, a wholesaler, a factory and a supplier. There is a lead time between each echelon in the supply chain. The question is how best to manage this supply chain. 

We can mode this setup with SupplyChainSimulation.jl as follows.

```julia
p = Single("product")

customer = Customer("customer")
retailer = Storage("retailer", Dict(p => 0.1))
wholesaler = Storage("wholesaler", Dict(p => 0.1))
factory = Storage("factory", Dict(p => 0.1))
supplier = Supplier("supplier")

horizon = 20

l = Lane(; origin = retailer, destination = customer, unit_cost = 0)
l2 = Lane(; origin = wholesaler, destination = retailer, unit_cost = 0, lead_time = 2)
l3 = Lane(; origin = factory, destination = wholesaler, unit_cost = 0, lead_time = 2)
l4 = Lane(; origin = supplier, destination = factory, unit_cost = 0, lead_time = 4)

policy2 = NetUptoOrderingPolicy(0)
policy3 = NetUptoOrderingPolicy(0)
policy4 = NetUptoOrderingPolicy(0)

network = Network([supplier], [retailer, wholesaler, factory], [customer], get_trips([l, l2, l3, l4], horizon), [p])

initial_states = [State(; on_hand_inventory = Dict(
                                                retailer => Dict(p => 20), 
                                                wholesaler => Dict(p => 20), 
                                                factory => Dict(p => 20)), 
                        demand = Dict((customer, p) => rand(Poisson(10), horizon)),
                        policies = Dict(
                                        (l2, p) => policy2,
                                        (l3, p) => policy3,
                                        (l4, p) => policy4)
                ) for i in 1:30]

optimize!(network, horizon, initial_states...)
```
The optimizer will then run and return the best policies.

If you run this code you will see that the policies do extremely well with no bullwhip effect. Inventory management solved? Not fully. Let's note that (1) the policies are tuned to a specific scenario (albeit stochastic) and (2) the optimizer optimizes across echelons (as if the whole supply chain is integrated). This is a best case scenario. Different setups can be tested. For example you can add more scenarios or you can change the policies to limit what they can see. Depending on the setup the bullwhip effect can be more or less strong. Being able to simulate these results is on key advantage of using SupplyChainSimulation.jl.
## API

```@autodocs
Modules = [SupplyChainSimulation]
Order   = [:type, :function]
```
