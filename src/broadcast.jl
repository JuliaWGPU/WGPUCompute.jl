# broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct WgpuArrayStyle{N} <: AbstractGPUArrayStyle{N} end

WgpuArrayStyle(::Val{N}) where N = WgpuArrayStyle{N}()
WgpuArrayStyle{M}(::Val{N}) where {N,M} = WgpuArrayStyle{N}()

BroadcastStyle(::Type{<:WgpuArray{T,N}}) where {T,N} = WgpuArrayStyle{N}()

Base.similar(bc::Broadcasted{WgpuArrayStyle{N}}, ::Type{T}, dims) where {N,T} =
    Base.similar(WgpuArray{T, length(dims)}, dims)
