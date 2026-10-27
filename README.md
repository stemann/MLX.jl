# MLX.jl

[![Documentation](https://img.shields.io/badge/documentation-dev-blue.svg)](https://stemann.github.io/MLX.jl/)
[![Build Status](https://github.com/stemann/MLX.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/stemann/MLX.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/stemann/MLX.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/stemann/MLX.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

A Julia API for [MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine learning, via [MLX C](https://github.com/ml-explore/mlx-c).

An `MLXArray` is an `AbstractArray` that can be used like an `Array`, but whose operations are computed lazily by MLX, on the CPU or on Apple silicon GPUs.

MLX.jl aims to provide two APIs:

- An `Array`-compatible API, where `Base` functions operate on `MLXArray`s like they do on `Array`, with some differences, e.g. operations may return `Float32` rather than `Float64` results, as MLX does not support `Float64` on GPUs.
- An MLX-compatible API, where `MLX.<function>`s follow MLX's semantics, e.g. in broadcasting and type promotion, to ease porting code written for MLX in other languages.

## Supported platforms

MLX.jl should support all of Julia's platforms, via the prebuilt binaries of [MLX](https://github.com/JuliaBinaryWrappers/MLX_jll.jl) and [MLX C](https://github.com/JuliaBinaryWrappers/MLX_C_jll.jl), but it is tested on the following platforms.

| **Operating system** | **Architecture** | **Devices** |
| --- | --- | --- |
| macOS | aarch64 | CPU, GPU (Metal) |
| macOS | x86_64 | CPU |
| Linux (glibc) | aarch64 | CPU |
| Linux (glibc) | x86_64 | CPU |

## Installation

```julia
using Pkg
Pkg.add("MLX")
```

## Usage

See the [quick start](https://stemann.github.io/MLX.jl/tutorials/quick-start/).
