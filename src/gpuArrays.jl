
# GPUArrays.device(x::WgpuArray) = x.device

gpuDevice = WGPUCore.getDefaultDevice()

struct WgpuDevice
	device::WGPUCore.GPUDevice
	id::Int
end

using GPUArrays

struct WgpuArrayBackend <: AbstractGPUBackend end

struct WgpuKernelContext <: AbstractKernelContext end


@inline function GPUArrays.launch_heuristic()

end

function GPUArrays.gpu_call(::WgpuArrayBackend, f, args, threads::Int, blocks::Int; name::Union{String, Nothing})
	@wgpu threads=threads grid=blocks name=name f(WgpuKernelContext(), args...)
end

# GPUArrays.blockidx()

GPUArrays.backend(::Type{<:WgpuArray}) = WgpuArrayBackend()

const GLOBAL_RNGs = Dict{WgpuDevice, GPUArrays.RNG}()

function GPUArrays.default_rng(::Type{<:WgpuArray})
    dev = WgpuDevice(gpuDevice, 1) # TODO choose device option
    get!(GLOBAL_RNGs, dev) do
        N = 128 # Size of default oneAPI working group with barrier, so should be good for Metal
        state = WgpuArray{NTuple{4, UInt32}}(undef, N)
        rng = GPUArrays.RNG(state)
        Random.seed!(rng)
        rng
    end
end

