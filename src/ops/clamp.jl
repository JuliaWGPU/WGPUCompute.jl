function clamp_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}, minval::T, maxval::T) where {T, N}
	gId = xDims.x*globalId.y + globalId.x
	value = x[gId]
	out[gId] = clamp(value, minval, maxval)
end


function clamp(x::WgpuArray{T, N}, minValue::T, maxValue::T) where {T, N}
	y = similar(x)
	@wgpukernel launch=true workgroupSizes=(4, 4) workgroupCount=(2, 2) clamp_kernel(x, y, minValue, maxValue)
	return y
end
