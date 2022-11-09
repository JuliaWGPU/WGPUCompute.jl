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


```julia

julia> using WGPUCompute

julia> y = WgpuArray((rand(4, 4, 1) .-0.5) .|> Float32)
4×4×1 WgpuArray{Float32, 3}:
[:, :, 1] =
 -0.383893   0.16837     0.0140184   0.199563
 -0.336961  -0.192162   -0.179518   -0.0335313
  0.444875   0.0344275  -0.100446    0.498892
 -0.354451  -0.488507   -0.078437   -0.132585

julia> @kernel function Relu(x::WgpuArray{T, N}) where {T, N}
               gIdx = globalId.x * globalId.y + globalId.z
               value = x[gIdx]
               out[gIdx] = max(value, 0.0)
       end
WARNING: Method definition Relu(WGPUCompute.WgpuArray{T, N}) where {T, N} in module Main at REPL[26]:1 overwritten on the same line.
Relu (generic function with 1 method)

julia> Relu(y)
4×4×1 WgpuArray{Float32, 3}:
[:, :, 1] =
 0.0       0.16837    0.0140184  0.199563
 0.0       0.0        0.0        0.0
 0.444875  0.0344275  0.0        0.498892
 0.0       0.0        0.0        0.0

> Internally compute shader is generated like below for Relu
 ┌ Info:
 │ struct IOArray {
 │     data:array<f32>
 │ };
 │
 │ @group(0) @binding(0) var<storage, read_write> input0:IOArray ;
 │ @group(0) @binding(1) var<storage, read_write> ouput1:IOArray ;
 │ @compute @workgroup_size(8, 8, 4)
 │ fn Relu(@builtin(global_invocation_id) global_id:vec3<u32>) {
 │     let gIdx = global_id.x * global_id.y+global_id.z;
 │     let value = input0.data[gIdx];
 │     ouput1.data[gIdx] = max(value, 0.0);
 │ }
 └

```

## Issues in example above.
- globalDim is not supported yet. As a temporary fix global_id.y is used.
- in generated shader code, @workgroup_size is hardcoded.

## Known issues

- Currently kernel macro has hardcoded sections and would expect :Relu as main function name. And it is
 being currently worked on.
- Doesn't have an api to pass @workgroup_size and @dispath_size yet. The example above is hardcoded.

