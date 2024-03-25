export naive_matmul_kernel, matmul

function naive_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gIdx = localId.x
	gIdy = localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = 0.0
	sum = 0.0
	for i in 0:xDims.y
		xIdx = xDims.x*i + gIdx
		yIdx = yDims.x*gIdy + i
		sum = sum + x[xIdx]*y[yIdx]
	end
	out[gId] = sum
end

function matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	aSize = size(x)
	bSize = size(y)
	@assert last(aSize) == first(bSize) 
	outSize = (first(aSize), last(bSize))
	@assert eltype(x) == eltype(y)
	out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	@wgpukernel(
		launch=true,
		workgroupSizes=outSize,
		workgroupCount=(1, 1),
		shmem=(),
		naive_matmul_kernel(x, y, out),
	)
	return out
end

function tiled_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gIdx = localId.x
	gIdy = localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = 0.0
	sum = 0.0
	for i in 0:xDims.y
		xIdx = xDims.x*i + gIdx
		yIdx = yDims.x*gIdy + i
		sum = sum + x[xIdx]*y[yIdx]
	end
	out[gId] = sum
end

function tiled_matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	aSize = size(x)
	bSize = size(y)
	@assert last(aSize) == first(bSize)
	outSize = (first(aSize), last(bSize))
	@assert eltype(x) == eltype(y)
	out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	wgSize = (16, 16)
	@wgpukernel(
		launch=true,
		workgroupSizes=outSize,
		workgroupCount=(1, 1),
		shmem=(:shmem1=>(Float32, (4, 4)), :shmem2=>(Float32, (4, 4))),
		tiled_matmul_kernel(x, y, out)
	)
	return out
end
