# WGPUCompute.jl
Array and kernel interface for WGPU

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
