using Revise
using WGPUCompute
using Test

empty!(task_local_storage())

function divfree_reduce_kernel(x::WgpuArray{T,N}, out::WgpuArray{T,N}, op::Function) where {T,N}
    gId = xDims.x * globalId.y + globalId.x
    W = Float32(xDims.x * xDims.y)
    steps = UInt32(ceil(log2(W)))
    out[gId] = x[gId]
    base = 2.0f0
    for itr in 0:steps
   		exponent = Float32(steps - itr - 1)
   		baseexp = pow(base, exponent)
		stride = UInt32(baseexp)
    	if localId.x < stride
			out[gId] = op(out[gId], out[gId + stride])
		end
	end
end

function divfree_reduce(x::WgpuArray{T,N}, op::Function) where {T,N}
    y = WgpuArray{T}(undef, size(x))
    @wgpukernel(
        launch = true,
        workgroupSizes = (8, 8),
        workgroupCount = (1, 1),
        shmem = (),
        divfree_reduce_kernel(x, y, op)
    )
    return (y |> collect)
end

x = WgpuArray{Float32}(rand(Float32, 8, 8))
z = divfree_reduce(x, +)

x_cpu = (x |> collect)

sum_cpu = sum(x_cpu)
sum_gpu = (z|>collect)[1]

@test sum_cpu ≈ sum_gpu
