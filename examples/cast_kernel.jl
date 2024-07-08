using WGPUCompute
using Test

function cast_kernel(x::WgpuArray{T, N}, out::WgpuArray{S, N}) where {T, S, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = S(ceil(x[gId]))
end

function cast(S::DataType, x::WgpuArray{T, N}) where {T, N}
	y = WgpuArray{S}(undef, size(x))
	@wgpukernel launch=true workgroupSizes=(4, 4) workgroupCount=(2, 2) shmem=() cast_kernel(x, y)
	return y
end

x = rand(Float32, 8, 8) .- 0.5f0

x_gpu = WgpuArray{Float32}(x)
z_gpu = cast(UInt32, x_gpu)
z_cpu = z_gpu |> collect

z = UInt32.(x .> 0.0)

@test z ≈ z_cpu

# TODO Bool cast is not working yet
# y = cast(Bool, x)
