using Test
using WGPUCompute
using WGPUCompute: elwise
x = WgpuArray{Float32, 2}(rand(16, 16))
y = elwise(+, x, x)

x_cpu = x |> collect
z_cpu = 2.0f0.*x_cpu

@test z_cpu ≈ (y |> collect)
