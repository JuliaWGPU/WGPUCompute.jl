using WGPUCompute
using WGPUCompute: tiled_matmul
using Chairmarks
using UnicodePlots
using Metal

logn = 4:12

for i in logn
	n = 2^(i)
	xCPU = rand(Float32, n, n)
	yCPU = rand(Float32, n, n)
	xGPU = WgpuArray{Float32, 2}(xCPU)
	yGPU = WgpuArray{Float32, 2}(yCPU)
	xMtl = MtlArray{Float32, 2}(xCPU)
	yMtl = MtlArray{Float32, 2}(yCPU)
	matmulBench = @b xMtl*yMtl
	tileBench = @b tiled_matmul(xGPU, yGPU)
	@info matmulBench.allocs matmulBench.time
	@info tileBench.allocs tileBench.time
end
