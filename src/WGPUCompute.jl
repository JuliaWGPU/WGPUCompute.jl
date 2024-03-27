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
using Tracy
using Infiltrator

# TODO enable debug based on flags 
# WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Debug)

include("array.jl")
include("compiler/execution.jl")
include("ops.jl")
include("broadcast.jl")
include("shaders.jl")

# array implementations
include("gpuArrays.jl")

end
