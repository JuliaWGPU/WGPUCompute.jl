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

function Relu(x::WgpuArray{T}) where T
	gIdx = globalId.x*globalId.y + globalId.z
	value = x[gIdx]
end

fexpr = @code_expr(Relu(y))

@capture(fexpr, function name_(args__) where Targs_ fbody__ end)

for stmts in fbody
	@info stmts
end

cntxt = emitWGSLJuliaBody(fbody, args)

wgpu(Relu, y) |> MacroTools.striplines

wgpu(Relu, y) |> MacroTools.flatten |> MacroTools.striplines

# using Debugger
# 
# @enter emitWGSLJuliaBody(fbody, args)
