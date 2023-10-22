using BenchmarkTools
using WGPUCompute
using Debugger

using WGPUCore

WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Debug)

aArray = WgpuArray{Float32}(undef, (1024, 1024, 10))
bArray = WgpuArray{Float32}(rand(Float32, (1024, 1024, 10)))

@benchmark copyto!(aArray, 1, bArray, 1, prod(size(aArray)))

