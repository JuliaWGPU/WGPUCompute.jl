using Revise
using WGPUCompute
using Test

empty!(task_local_storage())

function naive_reduce_kernel(x::WgpuArray{T,N}, out::WgpuArray{T,N}, op::Function) where {T,N}
    gId = xDims.x * globalId.y + globalId.x
    W = Float32(xDims.x * xDims.y)
    steps = UInt32(ceil(log2(W)))
    out[gId] = x[gId]
    base = 2.0f0
    for itr in 0:steps
	    if gId%2 == 0
    		exponent = Float32(itr)
    		baseexp = pow(base, exponent)
			stride = UInt32(baseexp)
			out[gId] = op(out[gId], out[gId + stride])
	    end
	end
end

function naive_reduce(x::WgpuArray{T,N}, *) where {T,N}
    y = WgpuArray{T}(undef, size(x))
    @wgpukernel(
        launch = true,
        workgroupSizes = (8, 8),
        workgroupCount = (1, 1),
        shmem = (),
        naive_reduce_kernel(x, y, *)
    )
    return (y |> collect)
end

x = WgpuArray{Float32}(rand(Float32, 8, 8))
z = naive_reduce(x, *)

x_cpu = (x |> collect)

sum_cpu = reduce(*, x_cpu)
sum_gpu = (z|>collect)[1]

@test sum_cpu ≈ sum_gpu
