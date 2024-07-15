
function broadcast_kernel(op::Function, x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = op(x[gId], y[gId])
end

function broadcast_kernel(op::Function, x::WgpuArray{T, N}, y::Float32, out::WgpuArray{T, N}) where {T, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = op(x[gId], y)
end

function broadcast_kernel(op::Function, x::Number, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = op(x[gId], y)
end

function elwise(f::Function, a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N}
	out = similar(a)
	@wgpukernel launch=true workgroupSizes=size(out) workgroupCount=(1, 1) shmem=() broadcast_kernel(f, a, b, out)
	return out
end

function elwise(f::Function, a::WgpuArray{T, N}, b::Number) where {T, N}
	out = similar(a)
	@wgpukernel launch=true workgroupSizes=size(out) workgroupCount=(1, 1) shmem=() broadcast_kernel(f, a, b, out)
	return out
end

function elwise(f::Function, a::Number, b::WgpuArray{T, N}) where {T, N}
	out = similar(a)
	@wgpukernel launch=true workgroupSizes=size(out) workgroupCount=(1, 1) shmem=() broadcast_kernel(f, a, b, out)
	return out
end

Base.:+(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N} = elwise(+, a, b)
Base.:-(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N} = elwise(-, a, b)
Base.:*(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N} = elwise(*, a, b)
Base.:/(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N} = elwise(/, a, b)
Base.:+(a::WgpuArray{T, N}, b::Number) where {T, N} = elwise(+, a, b)
Base.:-(a::WgpuArray{T, N}, b::Number) where {T, N} = elwise(-, a, b)
Base.:*(a::WgpuArray{T, N}, b::Number) where {T, N} = elwise(*, a, b)
Base.:/(a::WgpuArray{T, N}, b::Number) where {T, N} = elwise(/, a, b)
Base.:+(a::Number, b::WgpuArray{T, N}) where {T, N} = elwise(+, a, b)
Base.:-(a::Number, b::WgpuArray{T, N}) where {T, N} = elwise(-, a, b)
Base.:*(a::Number, b::WgpuArray{T, N}) where {T, N} = elwise(*, a, b)
Base.:/(a::Number, b::WgpuArray{T, N}) where {T, N} = elwise(/, a, b)
