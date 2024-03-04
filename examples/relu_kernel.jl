using WGPUCompute
using MacroTools

y = WgpuArray((rand(4, 4, 1) .-0.5) .|> Float32)

(@macroexpand @wgpukernel workgrouSizes=(4,4) workgroupCount=(2,2) function relu_kernel(
	x::WgpuArray{T, N},
	out::WgpuArray{T, N}
	) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, 0.0)
end) |> MacroTools.striplines

@wgpukernel workgrouSizes=(4,4) workgroupCount=(2,2) function relu_kernel(
	x::WgpuArray{T, N},
	out::WgpuArray{T, N}
	) where {T, N}
	gIdx = globalId.x * globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, zero())
end


function relu(x::WgpuArray{T, N}) where {T, N}
	y = similar(x)
	relu_kernel(x, y)
	return y
end

relu(y)

