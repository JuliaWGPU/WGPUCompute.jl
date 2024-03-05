using WGPUCompute

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
	@wgpukernel launch=true workgroupSizes=(4, 4) workgroupCount=(2, 2) cast_kernel(x, y)
	# previously this is how it has to be done
	#kobj = @macroexpand(@wgpukernel workgroupSizes=(4, 4) workgroupCount=(2, 2) cast_kernel(x, y))
	#kobj.kernelFunc(x, y)
	return y
end

x = WgpuArray{Float32}(rand(Float32, 8, 8) .- 0.5f0)
z = cast(UInt32, x)

# TODO Bool cast is not working yet
# y = cast(Bool, x)
