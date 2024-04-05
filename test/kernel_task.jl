using WGPUCore
using WGPUCompute
using WGPUCompute: getCurrentDevice
using FileIO

gpuDevice = getCurrentDevice()

shaderBytes = read(joinpath(ENV["HOME"], ".julia/dev/scratch.wgsl"))

shaderInfo = WGPUCore.loadWGSL(shaderBytes)

t = @task WGPUCore.createShaderModule(
		gpuDevice,
		"shaderCode",
		shaderInfo.shaderModuleDesc,
		nothing,
		nothing,
	)

