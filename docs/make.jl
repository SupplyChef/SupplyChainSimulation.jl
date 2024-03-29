import Pkg;

using Documenter
using SupplyChainSimulation

makedocs(
    sitename = "SupplyChainSimulation",
    format = Documenter.HTML(),
    modules = [SupplyChainSimulation],
    pages = ["index.md"]
)

deploydocs(;
    repo="https://github.com/SupplyChef/SupplyChainSimulation.jl",
    devbranch = "master"
)