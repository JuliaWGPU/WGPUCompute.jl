using MacroTools
using LinearAlgebra
using StaticArrays
using GeometryBasics: Mat4

export GPUManager, composeShader, addLayer!

abstract type AbstractLayer{T} end

function getWgpuDevice()
	get!(task_local_storage(), :WGPUDevice) do
		WGPU.getDefaultDevice()
	end
end

function composeShader(gpuDevice, layer, x; binding=3)
	src = quote end
	
	defaultSource = quote
		struct IOArray
			data::WArray{$(eltype(x))}
		end
	end
	
	push!(src.args, defaultSource)
	push!(src.args, getShaderCode(layer, x; binding = binding))
	
	try
		createShaderObj(gpuDevice, src; savefile=true)
	catch(e)
		@info e
		rethrow(e)
	end
end

struct LayerManager{Tin, Tout}
	layer::AbstractLayer{Tin}
end

function addLayer!(model, layer)
	push!(layers, layer)
	setup(layer)
end

function setup(layer)
	cshader = composeShader(gpuDevice, scene, object; binding=binding)
	scene.cshader = cshader
	@info cshader.src
	preparePipeline(gpuDevice, scene, object)
end

