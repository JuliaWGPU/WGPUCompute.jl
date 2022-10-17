module WGPUCompute

using WGPUCore
using WGPUNative
using GPUArrays
using GPUCompiler
using ExprTools
using LLVM
using Adapt
using Reexport
using WGSLTypes

include("state.jl")
include("initialization.jl")
include("memory.jl")
include("array.jl")
include("compiler/gpucompiler.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")
include("utilities.jl")
include("broadcast.jl")

include("shaders.jl")
include("layerManager.jl")

abstract type AbstractLayer{T} end 

for (root, dirs, files) in walkdir(joinpath(@__DIR__, "layers"))
	for file in files
		include(joinpath(root, file))
	end	
end

export ReLULayer
	
end
