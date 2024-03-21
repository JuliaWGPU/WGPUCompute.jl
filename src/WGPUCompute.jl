module WGPUCompute

using WGPUNative
using WGPUCore
using GPUArrays
using GPUCompiler
using ExprTools
using LLVM
using Adapt
using Reexport
using WGSLTypes

# TODO enable debug based on flags 
# WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Debug)

include("array.jl")
include("compiler/execution.jl")
include("broadcast.jl")
include("shaders.jl")

# array implementations
include("gpuArrays.jl")

abstract type AbstractLayer{T} end 

"""
for (root, dirs, files) in walkdir(joinpath(@__DIR__, "layers"))
	for file in files
		include(joinpath(root, file))
	end	
end
"""

end
