# WGPUCompute

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://arhik.github.io/WGPUCompute.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://arhik.github.io/WGPUCompute.jl/dev/)
[![Build Status](https://github.com/arhik/WGPUCompute.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/arhik/WGPUCompute.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/arhik/WGPUCompute.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/arhik/WGPUCompute.jl)

WIP progress

```julia 

using BenchmarkTools 
using WGPUCompute

aArray = WgpuArray{Float32}(undef, (1024, 1024, 100)) 
bArray = WgpuArray{Float32}(rand(Float32, (1024, 1024, 100)))

@benchmark copyto!(aArray, 1, bArray, 1, prod(size(aArray)))

```
```
BenchmarkTools.Trial: 403 samples with 1 evaluation.
 Range (min … max):  30.041 μs … 15.397 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     13.710 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   12.424 ms ±  4.124 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▄                                                   ▇█
  █▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▁▁▁████▇▅▆ ▆
  30 μs        Histogram: log(frequency) by time      15.3 ms <

 Memory estimate: 3.06 KiB, allocs estimate: 96.
 ```
