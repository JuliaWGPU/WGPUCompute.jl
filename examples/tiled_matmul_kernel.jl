using Revise
using WGPUCompute

using StaticArrays

const Vec2{T} = SVector{2, T}
const Vec3{T} = SVector{3, T}
const Vec4{T} = SVector{4, T}
const Mat2{T} = SMatrix{2, 2, T, 4}
const Mat3{T} = SMatrix{3, 3, T, 9}
const Mat4{T} = SMatrix{4, 4, T, 16}
const Vec{N, T} = SVector{N, T}


x = WgpuArray{Float32, 2}(rand(16, 16));
y = WgpuArray{Float32, 2}(rand(16, 16));

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
		shmem=(:shmem1=>(Vec4{Float32}, (4, 4)), :shmem2=>(Float32, (4, 4))),
		tiled_matmul_kernel(x, y, out)
	)
	return out
end

Base.:*(x::WgpuArray{T, N}, y::WgpuArray{T, N})  where {T, N} = tiled_matmul(x, y)

z = x*y
