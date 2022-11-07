using WGPUCompute

y = WgpuArray(rand(32, 32, 4) .|> Float32);

@kernel function Relu(x::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, 0.0)
end

Relu(y)

@macroexpand @kernel function Relu(x::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, 0.0)
end

@kernel Relu(x)
