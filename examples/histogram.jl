using Revise
using WGPUCompute
using Test

empty!(task_local_storage())

function histogram_kernel(hist::WgpuArray{WAtomic{T}, N1}, x::WgpuArray{T, N2}, iSize::UInt32) where {T, N1, N2}
	gId = workgroupId.x*workgroupId.y + localId.x
	stride = workgroupDims.x*workgroupCount.x
	while gId < iSize
		val = x[gId]
		hist[val] += T(1)
		gId += stride
	end
end

function histogram(x::WgpuArray{T, N1}, hist::WgpuArray{S, N2}) where {T, S, N1, N2}
	y = WgpuArray{WAtomic{UInt32}}(undef, nbins)
	copyto!(y, hist)
	@wgpukernel(
		launch=true,
		workgroupSizes=(64,),
		workgroupCount=(1,),
		shmem=(),
		histogram_kernel(y, x, reduce(*, size(x)) |> UInt32)
	)
	copyto!(hist, y)
	return hist
end

nbins = 10
x = WgpuArray{UInt32}(rand(UInt32, 64) .% nbins)
hist = WgpuArray{UInt32}(zeros(UInt32, 10))

z = histogram(x, hist)



