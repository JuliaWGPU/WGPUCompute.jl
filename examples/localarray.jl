using Revise
using WGPUCompute
using Test

function localarray_kernel(x::WgpuArray{T, N}, out::WgpuArray{S, N}) where {T, S, N}
	a = Vec4{Float32}(1.0f0, 2.0f0, 3.0f0, 4.0f0);
	gId = xDims.x*globalId.y + globalId.x
	out[gId] = S(ceil(x[gId]))
end

function localarray(S::DataType, x::WgpuArray{T, N}) where {T, N}
	y = WgpuArray{S}(undef, size(x))
	@wgpukernel launch = true workgroupSizes=(4, 4) workgroupCount=(2, 2) shmem=() localarray_kernel(x, y)
	return y
end

x = rand(Float32, 8, 8) .- 0.5f0

x_gpu = WgpuArray{Float32}(x)
z_gpu = localarray(UInt32, x_gpu)
z_cpu = z_gpu |> collect

z = UInt32.(x .> 0.0)

@test z ≈ z_cpu

