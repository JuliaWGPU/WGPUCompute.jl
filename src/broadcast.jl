# broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct WgpuArrayStyle{N} <: AbstractGPUArrayStyle{N} end
WgpuArrayStyle(::Val{N}) where N = WgpuArrayStyle{N}()
WgpuArrayStyle{M}(::Val{N}) where {N,M} = WgpuArrayStyle{N}()

BroadcastStyle(::Type{<:WgpuArray{T,N}}) where {T,N} = WgpuArrayStyle{N}()

Base.similar(bc::Broadcasted{WgpuArrayStyle{N}}, ::Type{T}) where {N,T} =
    similar(WgpuArray{T}, axes(bc))

Base.similar(bc::Broadcasted{WgpuArrayStyle{N}}, ::Type{T}, dims...) where {N,T} =
    WgpuArray{T}(undef, dims...)

# broadcasting type ctors isnt GPU compatible
Broadcast.broadcasted(::WgpuArrayStyle{N}, f::Type{T}, args...) where {N, T} =
    Broadcasted{WgpuArrayStyle{N}}((x...) -> T(x...), args, nothing)
