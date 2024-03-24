struct WgpuArrayBackend <: AbstractGPUBackend end

struct WgpuKernelContext <: AbstractKernelContext end

GPUArrays.backend(::Type{<:WgpuArray}) = WgpuArrayBackend()

GPUArrays.generic_matmatmul!(C::WgpuArray{T, N}, A::WgpuArray{T, N}, B::WgpuArray{T, N}, a::Bool, b::Bool) where {T, N} = begin
	@wgpukernel launch=true workgroupSizes=(size(C)) workgroupCount=(1, 1) naive_matmul_kernel(A, B, C)
	return C
end

@inline function GPUArrays.gpu_call(::WgpuArrayBackend, f::F, args::TT, threads::Int,
                                    blocks::Int; name::Union{String,Nothing}) where {F,TT}
    #@wgpukernel threads blocks name f(args...)
    f(args...)
end

