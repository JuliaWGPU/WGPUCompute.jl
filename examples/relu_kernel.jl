using WGPUCompute

y = WgpuArray((rand(4, 4) .-0.5) .|> Float32)

function relu_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gId = xDims.x*globalId.y + globalId.x
	value = x[gId]
	out[gId] = max(value, 0.0)
end

function relu(x::WgpuArray{T, N}) where {T, N}
	y = similar(x)
	@wgpukernel launch=true workgroupSizes=(4,4) workgroupCount=(1,1) shmem=() relu_kernel(x, y)
	return y
end

relu(y)

