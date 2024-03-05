using WGPUCompute
using MacroTools

y = WgpuArray((rand(4, 4) .-0.5) .|> Float32)

function relu_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	value = x[gId]
	out[gId] = max(value, 0.0)
end

function relu(x::WgpuArray{T, N}) where {T, N}
	y = similar(x)
	kobj = @wgpukernel workgrouSizes=(4,4) workgroupCount=(1,1) relu_kernel(x, y)
	kobj.kernelFunc(x, y)
	return y
end

relu(y)

