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

## Getting Started

## API

```@autodocs
Modules = [SupplyChainSimulation]
Order   = [:function, :type]
```