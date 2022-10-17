# Following Metal.jl->arrays.jl blindly
# https://github.com/JuliaGPU/Metal.jl/blob/main/src/array.jl
# Until I understand GPUArrays mechanics

using WGPUNative
using WGPUCore
using LinearAlgebra
using GPUArrays
using Adapt

export WgpuArray

# TODO MTL tracks cmdEncoder with task local storage. Thats neat.

function getCurrentDevice()
	get!(task_local_storage(), :WGPUDevice) do
		WGPUCore.getDefaultDevice()
	end
end

struct WgpuArrayPtr{T}
	buffer::WGPUCore.GPUBuffer
	offset::UInt
	function WgpuArrayPtr{T}(buffer::WGPUCore.GPUBuffer, offset=0) where {T}
		new(buffer, offset)
	end
end

Base.eltype(::Type{<:WgpuArrayPtr{T}}) where {T} = T

# TODO asserts for bound checks maybe ?
Base.:(+)(x::WgpuArrayPtr{T}, y::Integer) where {T} = WgpuArrayPtr{T}(x.buffer, x.offset + y)
Base.:(-)(x::WgpuArrayPtr{T}, y::Integer) where {T} = WgpuArrayPtr{T}(x.buffer, x.offset - y)
Base.:(+)(x::Integer, y::WgpuArrayPtr{T}) where {T} = WgpuArrayPtr{T}(x.buffer, y.offset + x)
Base.:(-)(x::Integer, y::WgpuArrayPtr{T}) where {T} = WgpuArrayPtr{T}(x.buffer, y.offset - x)

contents(ptr::WgpuArrayPtr{T}) where T = convert(Ptr{T}, contents(ptr.buffer) |> pointer) + ptr.offset
contents(buf::WGPUCore.GPUBuffer) = WGPUCore.mapRead(buf)

# following functions deviate WGPU api but not a concern since use case is different
# and if we are taking LLVM approach this should not matter

# GPU -> GPU
function Base.unsafe_copyto!(gpuDevice, dst::WgpuArrayPtr{T}, src::WgpuArrayPtr{T}, N::Integer) where T
	cmdEncoder = WGPUCore.createCommandEncoder(gpuDevice, "COMMAND ENCODER")
	WGPUCore.copyBufferToBuffer(
		cmdEncoder,
		src.buffer,
		src.offset |> Int,
		dst.buffer,
		dst.offset |> Int,
		N*sizeof(T)
	)
	WGPUCore.submit(gpuDevice.queue, [WGPUCore.finish(cmdEncoder),])
end

# GPU -> CPU
function Base.unsafe_copyto!(gpuDevice, dst::Ptr{T}, src::WgpuArrayPtr{T}, N::Integer) where T
	cmdEncoder = WGPUCore.createCommandEncoder(gpuDevice, "COMMAND ENCODER")
	# TODO we could simply readBuffer from WGPUCore.jl ?
	tmpBuffer = WGPUCore.createBuffer(
		" READ BUFFER TEMP ",
		gpuDevice,
		sizeof(T)*N,
		["CopyDst", "MapRead"],
		false
	)
	tmpWgpuArrayPtr = WgpuArrayPtr{T}(tmpBuffer, 0)
	Base.unsafe_copyto!(gpuDevice, tmpWgpuArrayPtr, src, N)
	WGPUCore.submit(gpuDevice.queue, [WGPUCore.finish(cmdEncoder),])
	unsafe_copyto!(dst, contents(tmpWgpuArrayPtr), N)
	WGPUCore.destroy(tmpBuffer)
end

# CPU -> GPU
function Base.unsafe_copyto!(gpuDevice, dst::WgpuArrayPtr{T}, src::Ptr{T}, N::Integer) where T
	cmdEncoder = WGPUCore.createCommandEncoder(gpuDevice, "COMMAND ENCODER")
	# TODO we could simply readBuffer from WGPUCore.jl ?
	# tmpBuffer = WGPUCore.createBuffer(
		# " READ BUFFER TEMP ",
		# gpuDevice,
		# N*sizeof(T),
		# ["CopyDst", "MapRead"],
		# false
	# )
	# Base.unsafe_copyto!(gpuDevice, tmpBuffer, src.buffer, N)
	# WGPUCore.submit(gpuDevice.queue, [WGPUCore.finish(cmdEncoder),])
	# unsafe_copy!(dst, WGPUCore.mapRead(tmpBuffer))
	WGPUCore.writeBuffer(gpuDevice.queue, dst.buffer, src)
end

function unsafe_fill!(gpuDevice, dst::WgpuArrayPtr{T}, value::Union{UInt8, Int8}, N::Integer) where T
	cmdEncoder = WGPUCore.createCommandEncoder(gpuDevice, "COMMAND ENCODER")
	WGPUCore.writeBuffer(gpuDevice.queue, dst.buffer, fill(value, N))
end


mutable struct WgpuArray{T, N} <: AbstractGPUArray{T, N}
	dims::Dims{N}
	storageData::Union{Vector{T}, Array{T}}
	maxSize::Int
	offset::Int
	storageBuffer::WGPUCore.GPUBuffer
	bindGroup::Union{Nothing, Int}
	computePipeline::Any # Dict ?
	# offset

	# TODO remove this when this is handled by xs::AbstractArray
	function WgpuArray{T, N}(data::Union{Vector, Array{T, N}}) where {T, N}
		device = getCurrentDevice()
		storageData = data[:]
		(storageBuffer, _) = WGPUCore.createBufferWithData(
			device,
			"WgpuArray Buffer",
			storageData,
			["Storage", "CopyDst", "CopySrc"]
		)
		bindGroup = nothing
		computePipeline = nothing
		obj = new{T, length(size(data))}(
			Dims(size(data)), 
			storageData,
			length(storageData)*sizeof(T),
			0,
			storageBuffer, 
			bindGroup, 		# TODO remove this coupling
			computePipeline # TODO remove this coupling
		)
		finalizer(obj) do arr
			obj = nothing
		end
	end
	
	function WgpuArray{T, N}(::UndefInitializer, dims::Dims{N}) where {T, N}
		Base.allocatedinline(T) || error("WgpuArray only supports element types that are stored inline ")
		maxSize = prod(dims) * sizeof(T)
		bufsize = if Base.isbitsunion(T)
			maxsize + prod(dims)
		else
			maxSize
		end

		dev = getCurrentDevice()
		
		if bufsize > 0
			storageData = Array{T}(undef, prod(dims))
			(storageBuffer, _) = WGPUCore.createBufferWithData(
				dev,
				"WgpuArray Buffer",
				storageData,
				["Storage", "CopyDst", "CopySrc"],
			)
		end
		bindGroup = nothing
		computePipeline = nothing
		obj = new(dims, storageData, maxSize, 0, storageBuffer, bindGroup, computePipeline)
		finalizer(obj) do arr
			obj = nothing
		end
		return obj
	end

	function WgpuArray{T, N}(buffer::WgpuArray, dims::Dims{T}) where {T, N}
		obj = new{T, N}(dims, storageData, prod(dims)*sizeof(T), 0, buffer, bindGroup, computePipeline)
		finalizer(obj) do arr
			obj = nothing
		end
		return obj
	end
	
end

Base.eltype(::Type{WgpuArray{T}}) where T = T
Base.eltype(::Type{WgpuArray{T, N}}) where {T, N} = T

# constructors (borrowed from WgpuArray for quick prototyping)
WgpuArray{T, N}(::UndefInitializer, dims::Integer...) where {T, N} = 
	WgpuArray{T, N}(undef, Dims(dims))

# type but not dimensionality specified 
WgpuArray{T}(::UndefInitializer, dims::Dims{N}) where {T, N} = WgpuArray{T, N}(undef, dims)
WgpuArray{T}(::UndefInitializer, dims::Integer...) where {T, N} = 
	WgpuArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructors
WgpuArray{T, 1}() where {T} = WgpuArray{T, 1}(undef, 0)

Base.similar(a::WgpuArray{T,N}) where {T,N} = WgpuArray{T,N}(undef, size(a))
Base.similar(a::WgpuArray{T}, dims::Base.Dims{N}) where {T,N} = WgpuArray{T,N}(undef, dims)
Base.similar(a::WgpuArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} =
  WgpuArray{T,N}(undef, dims)

function Base.copy(a::WgpuArray{T,N}) where {T,N}
  b = similar(a)
  @inbounds copyto!(b, a)
end


# array interface
Base.elsize(::Type{<:WgpuArray{T}}) where {T} = sizeof(T)
Base.size(x::WgpuArray) = x.dims
Base.sizeof(x::WgpuArray) = Base.elsize(x)*length(x)

Base.pointer(x::WgpuArray{T}) where {T} = Base.unsafe_convert(WgpuArrayPtr{T}, x)
@inline Base.pointer(x::WgpuArray{T}, i::Integer) where T = begin
	Base.unsafe_convert(WgpuArrayPtr{T}, x) + Base._memory_offset(x, i)
end

Base.unsafe_convert(t::Type{WgpuArrayPtr{T}}, x::WgpuArray{T}) where T = begin
	WgpuArrayPtr{T}(x.storageBuffer, x.offset*Base.elsize(x))
end

# interop with other arrays
@inline function WgpuArray{T,N}(xs::AbstractArray{T,N}) where {T,N}
  A = WgpuArray{T,N}(undef, size(xs))
  copyto!(A, convert(Array{T}, xs))
  return A
end

WgpuArray{T,N}(xs::AbstractArray{S,N}) where {T,N,S} = WgpuArray{T,N}(map(T, xs))

# underspecified constructors
WgpuArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = WgpuArray{T,N}(xs)
(::Type{WgpuArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = WgpuArray{S,N}(x)
WgpuArray(A::AbstractArray{T,N}) where {T,N} = WgpuArray{T,N}(A)

# idempotency
WgpuArray{T,N}(xs::WgpuArray{T,N}) where {T,N} = xs


## derived types

# wrapped arrays: can be used in kernels
const WrappedWgpuArray{T,N} = Union{WgpuArray{T,N}, WrappedArray{T,N,WgpuArray,WgpuArray{T,N}}}
const WrappedWgpuVector{T} = WrappedWgpuArray{T,1}
const WrappedWgpuMatrix{T} = WrappedWgpuArray{T,2}
const WrappedWgpuVecOrMat{T} = Union{WrappedWgpuVector{T}, WrappedWgpuMatrix{T}}


## conversions

Base.convert(::Type{T}, x::T) where T <: WgpuArray = x


## interop with C libraries

Base.unsafe_convert(::Type{<:Ptr}, x::WgpuArray) =
  throw(ArgumentError("cannot take the host address of a $(typeof(x))"))

Base.unsafe_convert(t::Type{WGPUCore.GPUBuffer}, x::WgpuArray) = x.buffer


## interop with CPU arrays

Base.unsafe_wrap(t::Type{<:Array}, arr::WgpuArray, dims; own=false) = unsafe_wrap(t, arr.buffer, dims; own=own)

# We dont convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{WgpuArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(WgpuArray, xs)

# if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:WgpuArray{T}}, xs::AbstractArray) where {T} =
  isbits(xs) ? xs : convert(WgpuArray{T}, xs)

Adapt.adapt_storage(::Type{Array}, xs::WgpuArray) = convert(Array, xs)

Base.collect(x::WgpuArray{T,N}) where {T,N} = copyto!(Array{T,N}(undef, size(x)), x)


# GPUArray

device(array::WgpuArray) = array.storageBuffer.device

## memory copying

function Base.copyto!(dest::WgpuArray{T}, doffs::Integer, src::Array{T}, soffs::Integer,
                      n::Integer) where T
  (n==0 || sizeof(T) == 0) && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(device(dest), dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::WgpuArray{T}, src::Array{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::Array{T}, doffs::Integer, src::WgpuArray{T}, soffs::Integer,
                      n::Integer) where T
  (n==0 || sizeof(T) == 0) && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(device(src), dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::Array{T}, src::WgpuArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::WgpuArray{T}, doffs::Integer, src::WgpuArray{T}, soffs::Integer,
                      n::Integer) where T
  (n==0 || sizeof(T) == 0) && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  # TODO: which device to use here?
  if device(dest) == device(src)
    unsafe_copyto!(device(dest), dest, doffs, src, soffs, n)
  else
    error("Copy between different devices not implemented")
  end
  return dest
end

Base.copyto!(dest::WgpuArray{T}, src::WgpuArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.unsafe_copyto!(dev, dest::WgpuArray{T}, doffs, src::Array{T}, soffs, n) where T
  # these copies are implemented using pure memcpys, not API calls, so arent ordered.
  # synchronize()

  GC.@preserve src dest unsafe_copyto!(dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dev, dest::Array{T}, doffs, src::WgpuArray{T}, soffs, n) where T
  # these copies are implemented using pure memcpys, not API calls, so arent ordered.
  # synchronize()

  GC.@preserve src dest unsafe_copyto!(dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dev, dest::WgpuArray{T}, doffs, src::WgpuArray{T}, soffs, n) where T
  # these copies are implemented using pure memcpys, not API calls, so arent ordered.
  # synchronize()

  GC.@preserve src dest unsafe_copyto!(dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end


# Base.show(array::WgpuArray) = Base.show(array.storageData)


## utilities

zeros(T::Type, dims...) = fill!(WgpuArray{T}(undef, dims...), 0)
ones(T::Type, dims...) = fill!(WgpuArray{T}(undef, dims...), 1)
zeros(dims...) = zeros(Float32, dims...)
ones(dims...) = ones(Float32, dims...)
fill(v, dims...) = fill!(WgpuArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(WgpuArray{typeof(v)}(undef, dims...), v)


device(a::SubArray) = device(parent(a))

# we dont really want an array, so dont call `adapt(Array, ...)`,
# but just want WgpuArray indices to get downloaded back to the CPU.
# this makes sure we preserve array-like containers, like Base.Slice.
struct BackToCPU end
Adapt.adapt_storage(::BackToCPU, xs::WgpuArray) = convert(Array, xs)

@inline function Base.view(A::WgpuArray, I::Vararg{Any,N}) where {N}
    J = to_indices(A, I)
    @boundscheck begin
        # Bases boundscheck accesses the indices, so make sure they reside on the CPU.
        # this is expensive, but it is a bounds check after all.
        J_cpu = map(j->adapt(BackToCPU(), j), J)
        checkbounds(A, J_cpu...)
    end
    J_gpu = map(j->adapt(WgpuArray, j), J)
    Base.unsafe_view(Base._maybe_reshape_parent(A, Base.index_ndims(J_gpu...)), J_gpu...)
end

# pointer conversions
## contiguous
function Base.unsafe_convert(::Type{WGPUCore.GPUBuffer}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P}
    return Base.unsafe_convert(WGPUCore.GPUBuffer, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end

## reshaped
function Base.unsafe_convert(::Type{WGPUCore.GPUBuffer}, V::SubArray{T,N,P,<:Tuple{Vararg{Union{Base.RangeIndex,Base.ReshapedUnitRange}}}}) where {T,N,P}
   return Base.unsafe_convert(WGPUCore.GPUBuffer, parent(V)) +
          (Base.first_index(V)-1)*sizeof(T)
end


## PermutedDimsArray

device(a::Base.PermutedDimsArray) = device(parent(a))

Base.unsafe_convert(::Type{WGPUCore.GPUBuffer}, A::PermutedDimsArray) where {T} =
    Base.unsafe_convert(WGPUCore.GPUBuffer, parent(A))


## reshape

function Base.reshape(a::WgpuArray{T,M}, dims::NTuple{N,Int}) where {T,N,M}
  if prod(dims) != length(a)
      throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(size(a))"))
  end

  if N == M && dims == size(a)
      return a
  end

  _derived_array(T, N, a, dims)
end

# create a derived array (reinterpreted or reshaped) that is still a WgpuArray
@inline function _derived_array(::Type{T}, N::Int, a::WgpuArray, osize::Dims) where {T}
  offset = (a.offset * Base.elsize(a)) ÷ sizeof(T)
  WgpuArray{T,N}(a.buffer, osize; a.maxsize, offset)
end


## reinterpret

device(a::Base.ReinterpretArray) = device(parent(a))

Base.unsafe_convert(::Type{WGPUCore.GPUBuffer}, a::Base.ReinterpretArray{T,N,S} where N) where {T,S} =
  WGPUCore.GPUBuffer(Base.unsafe_convert(ZePtr{S}, parent(a)))


## unsafe_wrap

function Base.unsafe_wrap(t::Type{<:Array{T}}, buf::WGPUCore.GPUBuffer, dims; own=false) where T
    ptr = convert(Ptr{T}, contents(buf))
    return unsafe_wrap(t, ptr, dims; own)
end

function Base.unsafe_wrap(t::Type{<:Array{T}}, ptr::WgpuArrayPtr{T}, dims; own=false) where T
    return unsafe_wrap(t, contents(ptr), dims; own)
end
