import Pkg;

using Documenter
using SupplyChainSimulation

makedocs(
    sitename = "SupplyChainSimulation",
    format = Documenter.HTML(),
    modules = [SupplyChainOptimization],
    pages = []
)

deploydocs(;
    repo="https://github.com/SupplyChef/SupplyChainSimulation.jl",
    devbranch = "master"
)