struct WgpuArrayBackend <: AbstractGPUBackend end

struct WgpuKernelContext <: AbstractKernelContext end

GPUArrays.backend(::Type{<:WgpuArray}) = WgpuArrayBackend()
