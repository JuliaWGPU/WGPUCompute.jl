using Revise
using WGPUCompute
using Test

function clamp_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}, minval::T, maxval::T) where {T, N}
    gId = xDims.x * globalId.y + globalId.x
    value = x[gId]
    out[gId] = clamp(value, minval, maxval)
end


function Base.clamp(x::WgpuArray{T, N}, minValue::T, maxValue::T) where {T, N}
    y = similar(x)
    @wgpukernel launch=true workgroupSizes=size(y) workgroupCount=(1, 1) shmem=() clamp_kernel(x, y, minValue, maxValue)
    return y
end

x = WgpuArray{Float32, 2}(rand(16, 16))

y = Base.clamp(x, 0.2f0, 0.5f0)
y_cpu = y |> collect

@testset "Clamp minimum and maximum" begin
	@test minimum(y_cpu) == 0.2f0
	@test maximum(y_cpu) == 0.5f0
end
