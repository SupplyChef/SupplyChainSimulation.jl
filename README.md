# SupplyChainSimulation.jl

[![Build status (Github Actions)](https://github.com/SupplyChef/SupplyChainSimulation.jl/workflows/CI/badge.svg)](https://github.com/SupplyChef/SupplyChainSimulation.jl/actions)

[![codecov.io](http://codecov.io/github/SupplyChef/SupplyChainSimulation.jl/coverage.svg?branch=master)](http://app.codecov.io/github/SupplyChef/SupplyChainSimulation.jl?branch=master)

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://SupplyChef.github.io/SupplyChainSimulation.jl/dev)

SupplyChainSimulation.jl is a package for modeling and simulating supply chains.

Learn more by reading the [documentation](https://SupplyChef.github.io/SupplyChainSimulation.jl/dev).

## Performance benchmarks

A scale/performance benchmark suite lives in `benchmark/` (separate from the correctness
test suite in `test/`). Run it with:

```
julia --project=benchmark benchmark/benchmarks.jl
```

Pass `--small` for a quick local smoke test with a much smaller synthetic network.
