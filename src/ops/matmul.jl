export naive_matmul_kernel, matmul

"""
	matmul_heuristics(x, y)
This function computes workgroup size and workgroup count heuristics for a given input.
This is used by `naive_matmul_kernel`.
"""
function matmul_heuristics(x, y)
	aSize = size(x)
	bSize = size(y)
	@assert last(aSize) == first(bSize) 
	outSize = (first(aSize), last(bSize))
	@assert eltype(x) == eltype(y)
	return (outSize, outSize, (1, 1))
end

"""
	naive_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
This is naive matrix multiplication implementation kernel. This is not supposed to be used as a regular
julia function. This needs to be passed to @wgpukernel to under transformations to `WGSL` compatible
shader code.
"""
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

"""
	matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
This is wrapper function for end users which uses naive implementation of matrix multiplication 
`naive_matmul_kernel` kernel for matrix computation. 
"""
function matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	(outSize, wgSize, wgCount) = matmul_heuristics(x, y)
	out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	@wgpukernel launch=true workgroupSizes=wgSize workgroupCount=wgCount shmem=() naive_matmul_kernel(x, y, out)
	return out
end

"""
	tiled_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
This is compute kernel which carries out tiled matrix multiplication of input `WgpuArrays`. This is 
not supposed to be used as a regular julia function. This instead needs to be passed to `@wgpukernel` macro
inside a wrapper function.
"""
function tiled_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	#set out matrix to zero
	gId = xDims.x*globalId.y + globalId.x
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

"""
	tiled_matmul_heuristics(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
This function computes workgroup size and workgroup count for a given input for
`tiled_matmul_heuristics` kernel function.
"""
function tiled_matmul_heuristics(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	aSize = size(x)
	bSize = size(y)
	@assert last(aSize) == first(bSize)
	outSize = (first(aSize), last(bSize))
	@assert eltype(x) == eltype(y)
	# For now valid only for square matrices of size powers of 2 and base size 16.
	wgSize = (16, 16) # This can be fixed for now
	wgCount = div.((outSize[1], outSize[2]), 16, RoundUp)
	return (outSize, wgSize, wgCount)
end

"""
	tiled_matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
This is user end matrix multiplication function which carries out tiled matrix multiplication of
input `WgpuArray` arguments.
"""
function tiled_matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	(outSize, wgSize, wgCount) = tiled_matmul_heuristics(x, y)
	out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	@wgpukernel(
		launch=true,
		workgroupSizes=wgSize,
		workgroupCount=wgCount,
		shmem=(:shmem1=>(Float32, wgSize), :shmem2=>(Float32, wgSize)),
		tiled_matmul_kernel(x, y, out)
	)
	return out
end

# Tiled is efficient among two ... more implementations need to evaluated
Base.:*(x::WgpuArray{T, N}, y::WgpuArray{T, N})  where {T, N} = tiled_matmul(x, y)

