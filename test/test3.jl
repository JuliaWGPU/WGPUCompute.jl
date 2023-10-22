using Revise
using WGPUCompute
using Debugger
using WGPUCore
using MacroTools
using CodeTracking

using WGPUCompute: getShaderCode

WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)

using WGSLTypes

y = WgpuArray(rand(256, 256, 32) .- 0.5 .|> Float32);

relu = ReLULayer{Float32}()

src = MacroTools.striplines(getShaderCode(relu, y))

dump(src)

function Relu(x::WgpuArray{T, N}) where {T, N}
	gIdx = globalId.x*globalId.y + globalId.z
	value = x[gIdx]
	out[gIdx] = max(value, 0.0)
end

fexpr = @code_expr(Relu(y))

@capture(fexpr, function name_(args__) where Targs__ fbody__ end)

for stmts in fbody
	@info stmts
end

cntxt = emitWGSLJuliaBody(fbody, args)

getShaderCode(relu, y) |> MacroTools.striplines

getShaderCode(relu, y) |> MacroTools.flatten |> MacroTools.striplines

# using Debugger
# 
# @enter emitWGSLJuliaBody(fbody, args)

