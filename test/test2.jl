using BenchmarkTools
using WGPUCompute
using Debugger
using WGPU
WGPU.SetLogLevel(WGPU.WGPULogLevel_Off)

using WGSLTypes

x = WgpuArray{Float32}(undef, (8, 8, 4))
y = WgpuArray(rand(8, 8, 4) .- 0.5 .|> Float32)

relu = ReLULayer{Float32}()
relu(y)
relu(x)

@benchmark z = relu(y)

