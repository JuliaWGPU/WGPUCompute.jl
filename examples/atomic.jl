using Revise
using WGPUCompute
using Test

# Silly example
# Need to comeup with better example

empty!(task_local_storage())

function atomiccount_kernel(hist::WgpuArray{T, N1}, x::WgpuArray{T, N2}, iSize::UInt32) where {T, N1, N2}
	gId = workgroupId.x*workgroupId.y + localId.x
	stride = workgroupDims.x*workgroupCount.x
	@wgpuatomic a::UInt32
	val = x[gId]
	a = hist[val]
	while gId < iSize
		val = x[gId]
		a += T(1)
		gId += stride
	end
	hist[val] = a
end

function atomiccount(x::WgpuArray{T, N1}, hist::WgpuArray{S, N2}) where {T, S, N1, N2}
	y = WgpuArray{UInt32}(undef, nbins)
	copyto!(y, hist)
	@wgpukernel(
		launch=true,
		workgroupSizes=(64,),
		workgroupCount=(1,),
		shmem=(),
		atomiccount_kernel(y, x, reduce(*, size(x)) |> UInt32)
	)
	return y
end

nbins = 10
x = WgpuArray{UInt32}(rand(UInt32, 64) .% nbins)
count = WgpuArray{UInt32}(zeros(UInt32, 10))

z = atomiccount(x, count)

# histogram(x)
