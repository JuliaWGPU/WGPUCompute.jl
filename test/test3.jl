using WGPUCompute
using Debugger
using WGPUCore
using MacroTools

using WGPUCompute: getShaderCode

WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)

using WGSLTypes

y = WgpuArray(rand(256, 256, 32) .- 0.5 .|> Float32);

relu = ReLULayer{Float32}()

src = MacroTools.striplines(getShaderCode(relu, y))

dump(src)

wgpu(relu, y)


function Relu(x::WgpuArray{T}) where T
	max(0.0, x)
end

dump(@code_expr(Relu(x)))
