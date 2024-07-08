using Revise
using WGPUCompute

x = WgpuArray{UInt32}(rand(UInt32, 10, 10))
y = WgpuArray{WAtomic{UInt32}}(undef, 10, 10)

cntxt = WGPUCompute.WgpuKernelContext()
Base.unsafe_copyto!(WGPUCompute.device(y), pointer(y, 1), pointer(x, 1), reduce(*, size(x)))

copyto!(y, x)

copyto!(x, y)

