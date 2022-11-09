using WGPUCompute
using MacroTools

y = WgpuArray((rand(4, 4, 1) .-0.5) .|> Float32)

(@macroexpand @kernel function Relu(x::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, 0.0)
end) |> MacroTools.striplines

@kernel function Relu(x::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = sin(2.0)
end

Relu(y)

# TODO this version should also be useful but may be not
# @kernel Relu(x)
