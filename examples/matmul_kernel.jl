using Revise
using WGPUCompute 
using Infiltrator
using Test

wgSize = 256 # TODO probe for this information

function matmul_heuristics(x, y)
	aSize = size(x)
	bSize = size(y)
	@assert last(aSize) == first(bSize) 
	outSize = (first(aSize), last(bSize))
	@assert eltype(x) == eltype(y)
	return (outSize, outSize, (1, 1))
end

function naive_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x
	gIdy = globalId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = 0.0
	sum = 0.0
	for i in 0:(xDims.y)
		xIdx = xDims.x*i + gIdx
		yIdx = yDims.x*gIdy + i
		sum = sum + x[xIdx]*y[yIdx]
	end
	out[gId] = sum
end

function matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	(outSize, wgSize, wgCount) = matmul_heuristics(x, y)
	out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	@wgpukernel launch=true workgroupSizes=wgSize workgroupCount=wgCount shmem=() naive_matmul_kernel(x, y, out)
	return out
end

xcpu = rand(Float32, 16, 20)
ycpu = rand(Float32, 20, 16)

x = WgpuArray{Float32, 2}(xcpu)

y = WgpuArray{Float32, 2}(ycpu)

out = matmul(x, y)

xcpu*ycpu

@test (xcpu*ycpu) ≈ (out |> collect)
