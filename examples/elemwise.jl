using WGPUCompute
using WGPUCompute: elwise
x = WgpuArray{Float32, 2}(rand(16, 16))
y = elwise(+, x, x)
