export naive_matmul_kernel, matmul

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


function tiled_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	lIdx = localId.x
	lIdy = localId.y
	gIdx = globalId.x
	gIdy = globalId.y
	
	#set out matrix to zero
	gId = xDims.x*gIdy + gIdx
	out[gId] = 0.0
	
	# set local variable = 0.0
	sum = 0.0
	
	for tileId in 0:numWorkgroups.y
		# copy block from x to shared memory
		xId = workgroupId.x*workgroupDims.x + localId.x
		yId = tileId*workgroupDims.y + localId.y
		sId = localId.y*workgroupDims.x + localId.x
		shmem1[sId] = x[yId*xDims.x + xId]
		
		# copy block from y to shared memory
		xId = tileId*workgroupDims.x + localId.x
		yId = workgroupId.y*workgroupDims.y + localId.y
		shmem2[sId] = y[yId*yDims.x + xId]
		synchronize()
		
		# block sums for each tid
		for i in 0:xDims.y/numWorkgroups.y
			sum = sum + shmem1[i*workgroupDims.x + localId.x]*shmem2[localId.y*workgroupDims.x + i]
		end
		synchronize()
	end
	
	out[gId] = sum
end
# For now valid only for square matrices of size powers of 2 and base size 16.
function tiled_matmul_heuristics(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	aSize = size(x)
	bSize = size(y)
	@assert last(aSize) == first(bSize)
	outSize = (first(aSize), last(bSize))
	@assert eltype(x) == eltype(y)
	wgSize = (16, 16) # This can be fixed for now
	wgCount = div.((outSize[1], outSize[2]), 16, RoundUp)
	return (outSize, wgSize, wgCount)
end

function tiled_matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	(outSize, wgSize, wgCount) = tiled_matmul_heuristics(x, y)
	@tracepoint "out alloc" out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	@tracepoint "kernel" @wgpukernel(
		launch=true,
		workgroupSizes=wgSize,
		workgroupCount=wgCount,
		shmem=(:shmem1=>(Float32, wgSize), :shmem2=>(Float32, wgSize)),
		tiled_matmul_kernel(x, y, out)
	)
	return out
end

