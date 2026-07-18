#=
Instantiates the shared Julia environment for every script under
examples/usecases/. Run once per CI job (or once locally) before any
model.jl / generate_post.jl:

    julia examples/usecases/setup.jl

Pins SupplyChainModeling to the same `main`-branch revision the package's
own CI uses (see ../../Project.toml's [sources] section and
../../.github/workflows/ci.yml) before dev-installing this repo's own
package, so Pkg resolves against that dev'd copy instead of re-resolving
SupplyChainModeling from the registry.
=#
using Pkg

Pkg.activate(@__DIR__)
Pkg.add(url="https://github.com/SupplyChef/SupplyChainModeling.jl", rev="main")
Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
Pkg.add(["Distributions", "JSON3"])
Pkg.instantiate()
