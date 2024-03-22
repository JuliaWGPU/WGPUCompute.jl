using Revise
using WGPUCompute 
using Infiltrator

function naive_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gIdx = localId.x
	gIdy = localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = 0.0
	sum = 0.0
	for i in 0:(xDims.x - 1)
		xIdx = xDims.x*gIdy + i
		yIdx = i*yDims.x + gIdx
		sum = sum + x[xIdx]*y[yIdx]
	end
	out[gId] = sum
end

function matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	aSize = size(x)
	bSize = size(y)
	@assert last(aSize) == first(bSize) 
	# TODO generalize
	outSize = (first(aSize), last(bSize))
	@assert eltype(x) == eltype(y)
	# TODO promote types
	out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	@wgpukernel launch=true workgroupSizes=outSize workgroupCount=(1, 1) naive_matmul_kernel(x, y, out)
	return out
end

xcpu = rand(Float32, 16, 32)
ycpu = rand(Float32, 32, 8)

x = WgpuArray{Float32, 2}(xcpu)

y = WgpuArray{Float32, 2}(ycpu)

matmul(x, y)

xcpu*ycpu
