module WGPUCompute

using WGPU
using WGPU_jll
using GPUArrays
using GPUCompiler
using SPIRV_LLVM_Translator_jll
using ExprTools
using LLVM
using Adapt
using Reexport

# include("state.jl")
# include("initialization.jl")
# include("memory.jl")
include("array.jl")
# include("compiler/gpucompiler.jl")
# include("compiler/execution.jl")
# include("compiler/reflection.jl")
# include("utilities.jl")
include("broadcast.jl")

end
