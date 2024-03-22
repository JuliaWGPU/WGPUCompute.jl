function mat_transpose(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gIdx = localId.x
	gIdy = localId.y
	outgId = xDims.y*gIdy + gIdx
	ingId = xDims.x*gIdx + gIdy
	out[outgId] = x[ingId]
end

function transpose(x::WgpuArray{T, N}) where {T, N}
	outSize = size(x) |> reverse
	out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	@wgpukernel launch=true workgroupSizes=outSize workgroupCount=(1, 1) mat_transpose(x, out)
	return out
end

