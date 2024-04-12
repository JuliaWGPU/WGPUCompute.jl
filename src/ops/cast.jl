export cast_kernel, cast

"""
	cast_kernel(x::WgpuArray{T, N}, out::WgpuArray{S, N}) where {T, S, N}
This is a compute kernel which casts the `x` array of eltype `T` to eltype `S`.
Users are not supposed to use this function call from julia. This instead needs to 
be wrapped with an additional function which uses `@wgpukernel` macro call to
convert the julia function definition to a equivalent `WGPU` kernel function.
"""
function cast_kernel(x::WgpuArray{T, N}, out::WgpuArray{S, N}) where {T, S, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = S(ceil(x[gId]))
end	

"""
	cast(S::DataType, x::WgpuArray{T, N}) where {T, N}
This is a wrapper function for `cast_kernel` kernel function. This is meant 
for users to `cast` from regular julia functions.
"""
function cast(S::DataType, x::WgpuArray{T, N}) where {T, N}
	y = WgpuArray{S}(undef, size(x))
	@wgpukernel launch=true workgroupSizes=(4, 4) workgroupCount=(2, 2) shmem=() cast_kernel(x, y)
	return y
end	
