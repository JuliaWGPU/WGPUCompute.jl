using Revise
using WGPUCompute 

function naive_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gId = xDims.x*globalId.y + globalId.x
	for i in 1:10
		stride_lhs = 4
		stride_rhs = 8
		x[gId] = T(stride_lhs * stride_rhs)
	end
end

function matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	aSize = size(x)
	bSize = size(y)
	@assert last(aSize) == first(bSize) # TODO generalize
	outSize = (first(aSize), last(bSize))
	@assert eltype(x) == eltype(y)
	# TODO promote types
	out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	@wgpukernel launch=true workgroupSizes=(4, 4) workgroupCount=(1, 1) naive_matmul_kernel(x, y, out)
	return out
end

x = WgpuArray{Float32, 2}(undef, 4, 4)

y = WgpuArray{Float32, 2}(undef, 4, 4)

matmul(x, y)

