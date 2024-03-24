
function broadcast_kernel(op::Function, x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = op(x[gId], y[gId])
end

function elwise(f::Function, a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N}
	out = similar(a)
	@wgpukernel launch=true workgroupSizes=size(out) workgroupCount=(1, 1) broadcast_kernel(f, a, b, out)
	return out
end

Base.:+(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N} = elwise(+, a, b)
Base.:-(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N} = elwise(-, a, b)
Base.:*(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N} = elwise(*, a, b)
Base.:/(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N} = elwise(/, a, b)


using Revise
using WGPUCompute
x = WgpuArray{Float32, 2}(rand(16, 16))

