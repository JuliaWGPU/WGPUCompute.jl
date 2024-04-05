using WGPUCore
using Reexport

export createShaderObj

struct ShaderObj
	src
	internal
	info
end

function createShaderObj(gpuDevice, shaderSource; savefile=false, debug = false)
	shaderSource = shaderSource |> wgslCode 
	@info shaderSource
	shaderBytes  = shaderSource |> Vector{UInt8}

	shaderInfo = WGPUCore.loadWGSL(shaderBytes)

	shaderObj = ShaderObj(
		shaderSource,
		WGPUCore.createShaderModule(
			gpuDevice,
			"shaderCode",
			shaderInfo.shaderModuleDesc,
			nothing,
			nothing
		) |> Ref,
		shaderInfo
	)

	if shaderObj.internal[].internal[] == Ptr{Nothing}()
		@error "Shader Obj creation failed"
		@info "Dumping shader to scratch.wgsl for later inspection"
		file = open("scratch.wgsl", "w")
		write(file, shaderSource)
		close(file)
		try
			run(`naga scratch.wgsl`)
		catch(e)
			@info shaderSource
			rethrow(e)
		end
	end

	return shaderObj
end

