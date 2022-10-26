using BenchmarkTools
using WGPUCompute
using Debugger
using WGPUCore

WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)

using WGSLTypes

x = WgpuArray{Float32}(undef, (256, 256, 256))
y = WgpuArray(rand(256, 256, 32) .- 0.5 .|> Float32)

relu = ReLULayer{Float32}()
relu(y)
relu(x)

@btime relu(y)

@benchmark relu(y)
