using Revise
using WGPUCompute
using Test

empty!(task_local_storage())

function naive_prefix_scan_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}, partials::WgpuArray{T, N}) where {T, N}
	gId = xDims.x * globalId.y + globalId.x
	W = Float32(xDims.x * xDims.y)
	steps = UInt32(ceil(log2(W)))
	out[gId] = x[gId]
	base = 2.0f0
	for itr in 0:steps
		v = 0.0f0
		exponent = Float32(itr)
		baseexp = pow(base, exponent)
		stride = UInt32(baseexp)
		if localId.x >= stride
			v = out[gId - stride]
		end
		synchronize()
		if localId.x >= stride
			out[gId] += v
		end
		synchronize()
	end

	if localId.x == workgroupDims.x - 1
		partials[workgroupId.x] = out[gId]
	end
end

function naive_prefix_partials_scatter_kernel(y::WgpuArray{T, N}, p::WgpuArray{T, N}) where {T, N}
	gId = yDims.x * globalId.y + globalId.x
	y[gId] += p[workgroupId.x - 1]
end

function prefix_scan_heuristics(x::WgpuArray{T, N}) where {T, N}
	div(reduce(*, size(x)), 256)
end

function naive_prefix_scan(x::WgpuArray{T, N}) where {T, N}
	y = similar(x)
	maxthreads = 256
	wgsize = div(reduce(*, size(x)), maxthreads)
	p = WgpuArray{T, N}(zeros(wgsize))
	@wgpukernel(
		launch=true,
		workgroupSizes = (maxthreads,),
		workgroupCount = (wgsize,),
		shmem = (),
		naive_prefix_scan_kernel(x, y, p)
	)
	pscan = cumsum(p |> collect)
	partials = WgpuArray{T, N}(pscan)
	@wgpukernel(
		launch=true,
		workgroupSizes = (maxthreads,),
		workgroupCount = (wgsize,),
		shmem = (),
		naive_prefix_partials_scatter_kernel(y, partials)
	)
	return y
end

x = WgpuArray{Float32}(rand(Float32, 2^22))
z = naive_prefix_scan(x,)

x_cpu = (x |> collect)
cumcpu = cumsum(x_cpu, dims=1)
cumgpu = (z |> collect)

@test all(x-> x < 10-6, cumcpu .- cumgpu)

