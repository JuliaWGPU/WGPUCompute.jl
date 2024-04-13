export clamp_kernel, clamp

"""
	clamp_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}, minVal::T, maxval::T) where {T, N}
This is a clamp compute kernel which takes input `x` and an uninitialized output `out` WgpuArrays,
along with clamp lower bound and upper bound values `minVal` and `maxVal` of type `T`. End users are not
supposed to call this function like regular julia function. This is instead needs to passed to `@wgpukernel`
macro to under go transformations into `WGSL` shader code.
"""

function clamp_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}, minval::T, maxval::T) where {T, N}
	gId = xDims.x*globalId.y + globalId.x
	value = x[gId]
	out[gId] = clamp(value, minval, maxval)
end

"""
	clamp(x::WgpuArray{T, N}, minValue::T, maxValue::T) where {T, N}
This is a clamp operator which takes `WgpuArray` as an input along with lower bound and upper bound clamp
values to clamp the input array to these bounds
"""
function clamp(x::WgpuArray{T, N}, minValue::T, maxValue::T) where {T, N}
	y = similar(x)
	@wgpukernel launch=true workgroupSizes=(4, 4) workgroupCount=(2, 2) shmem=() clamp_kernel(x, y, minValue, maxValue)
	return y
end
