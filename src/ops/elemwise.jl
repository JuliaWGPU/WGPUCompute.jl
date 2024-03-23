export broadcast_kernel, elemwise

function broadcast_kernel_gen(f::Function, a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N}
	:(function broadcast_kernel(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N}
		xdim = workgroupDims.x
		ydim = workgroupDims.y
		gIdx = workgroupId.x*xdim + localId.x
		gIdy = workgroupId.y*ydim + localId.y
		gId = xDims.x*gIdy + gIdx
		out[gIdx] = $f(x[gIdx], y[gIdx])
	end)
end

function elemwise(f::Function, a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N}
	broadcast_kernel = eval(@__MODULE__, broadcast_kernel_gen(f, a, b))
	out = similar(a)
	@wgpukernel launch=true workgroupSizes=size(out) workgroupCount=(1, 1) broadcast_kernel(a, b, out)
	return out	
end

Base.:+(a::WgpuArray{T, N}, b::WgpuArray{T, N}) where {T, N} = begin
	elemwise(+, a, b)
end

