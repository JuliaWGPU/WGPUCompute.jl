using WGPUCompute
using Debugger
using WGPUCore
using MacroTools
using CodeTracking
using Revise

using WGPUCompute: getShaderCode

WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)

using WGSLTypes

y = WgpuArray(rand(256, 256, 32) .- 0.5 .|> Float32);

relu = ReLULayer{Float32}()

src = MacroTools.striplines(getShaderCode(relu, y))

dump(src)

wgpu(relu, y)

function Relu(x::WgpuArray{T}) where T
	max_x = max(0.0, x)
	max_x += 2
	out = max_x*2
end

fexpr = @code_expr(Relu(y))

@capture(fexpr, function name_(args__) where Targs_ fbody__ end)


for stmts in fbody
	stmts
end


