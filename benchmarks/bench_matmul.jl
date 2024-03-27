using WGPUCompute
using WGPUCompute: tiled_matmul
using Chairmarks
using UnicodePlots

logn = 1:4

for i in logn
	n = 2^(i) 
	xCpu = rand(Float32, n, n)
	yCpu = rand(Float32, n, n)
	xGpu = WgpuArray{Float32, 2}(xCpu)
	yGpu = WgpuArray{Float32, 2}(yCpu)
	matmulBench = @b matmul(xGpu, yGpu)
	tileBench = @b tiled_matmul(xGpu, yGpu)
	@info matmulBench.allocs matmulBench.time
	@info tileBench.allocs tileBench.time
end
	
