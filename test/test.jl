using BenchmarkTools
using WGPUCompute
using Debugger
using WGPU
WGPU.SetLogLevel(WGPU.WGPULogLevel_Debug)
# 
# aArray = WgpuArray{Float32}(undef, (1024, 1024, 100))
# bArray = WgpuArray{Float32}(rand(Float32, (1024, 1024, 100)))
# 
# @benchmark copyto!(aArray, 1, bArray, 1, prod(size(aArray)))

using WGSLTypes

x = WgpuArray{Float32}(undef, (8, 8, 4))
y = WgpuArray(rand(8, 8, 4) .- 0.5 .|> Float32)

relu = ReLULayer{Float32}()
relu(y)
relu(x)

@benchmark relu(y)
