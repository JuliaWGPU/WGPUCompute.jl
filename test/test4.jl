using WGPUCompute

y = WgpuArray((rand(32, 32, 4) .-0.5) .|> Float32);

@macroexpand @kernel function Relu(x::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, 0.0)
end

@kernel function Relu(x::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, 0.0)
end

Relu(y)



# TODO this version should also be useful but may be not
# @kernel Relu(x)
