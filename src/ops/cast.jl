export cast_kernel, cast

function cast_kernel(x::WgpuArray{T, N}, out::WgpuArray{S, N}) where {T, S, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = S(ceil(x[gId]))
end	

function cast(S::DataType, x::WgpuArray{T, N}) where {T, N}
	y = WgpuArray{S}(undef, size(x))
	@wgpukernel launch=true workgroupSizes=(4, 4) workgroupCount=(2, 2) shmem=() cast_kernel(x, y)
	return y
end	

