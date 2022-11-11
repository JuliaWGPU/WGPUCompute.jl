using SPIRV
using Test

function test(gIndex, y, x)
	sin(x[gIndex.x])
end

interp = SPIRVInterpreter([INTRINSICS_GLSL_METHOD_TABLE])

target = @target interp test(::Vec{3, UInt32}, ::Vec{4, Float32}, ::Vec{4, Float32})

ir = compile(target, AllSupported())

interface = ShaderInterface(
	SPIRV.ExecutionModelGLCompute,
	# [SPIRV.StorageClassFunction, SPIRV.StorageClassInput],
	[SPIRV.StorageClassInput, SPIRV.StorageClassOutput, SPIRV.StorageClassInput,],
	SPIRV.dictionary([
		# 1 => Decorations(SPIRV.DecorationBuiltIn, SPIRV.BuiltInWorkgroupSize),
		1 => Decorations(SPIRV.DecorationBuiltIn, SPIRV.BuiltInGlobalInvocationId),
		2 => Decorations(SPIRV.DecorationLocation, 0 |> UInt32),
		3 => Decorations(SPIRV.DecorationLocation, 1 |> UInt32)
	])
)

shader = Shader(target, interface)

@test unwrap(validate(shader))

validate(shader)
