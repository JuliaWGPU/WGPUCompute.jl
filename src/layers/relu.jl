using WGSLTypes
using WGPU
using LinearAlgebra
using StaticArrays


abstract type AbstractLayer{T} end 
abstract type ActivationLayer{T} <: AbstractLayer{T} end


struct ReLULayer{T} <: ActivationLayer{T}
end

function compileShader(relu::ReLULayer{T}, x::AbstractArray{T}) where T
	shaderSrc = getShaderCode(relu, x)
	cShader = nothing
	try
		cShader = createShaderObj(getWgpuDevice(), shaderSrc; savefile = true)
	catch(e)
		@info e
		rethrow(e)
	end
	task_local_storage((:relu, :shader, T, size(x)), cShader)
	return cShader
end

function (relu::ReLULayer{T})(x::AbstractArray{T}) where T 
	dims = size(x)
	# get!(task_local_storage(), (:relu, T, size(x))) do
		# preparePipeline(relu, x)
	# end
	preparePipeline(relu, x)
	compute(
		task_local_storage((:relu, :pipeline, T, size(x))),
		task_local_storage((:relu, :bindgroup, T, size(x))),
		relu,
		x
	)
	return x
end


function getShaderCode(activation::ReLULayer{T}, x::AbstractArray{T}) where T
	shaderSrc = quote
		struct IOArray
			data::WArray{$T}
		end

		struct IOArrayVector
			data::WArray{Vec4{$T}}
		end

		struct IOArrayMatrix
			data::WArray{Mat4{$T}}
		end

		@var StorageReadWrite 0 0 input0::IOArray
		@var StorageReadWrite 0 1 output0::IOArray

		@compute @workgroupSize($(size(x)...)) function main(@builtin global_invocation_id => global_id::Vec3{UInt32})
			@let gIdx = global_id.x*global_id.y + global_id.z;
			@let value = input0.data[gIdx]
			output0.data[gIdx] = max(value, $(zero(eltype(activation))))
		end
		
	end
end


function getBindingLayouts(relu::ReLULayer; binding=0)
	bindingLayouts = [
		WGPU.WGPUBufferEntry => [
			:binding => binding,
			:visibility => "Compute",
			:type => "Storage"
		],
		WGPU.WGPUBufferEntry => [
			:binding => binding + 1,
			:visibility => "Compute",
			:type => "Storage"
		]
	]
end


function getBindings(relu::ReLULayer, x; binding=0)
	bindings = [
		WGPU.GPUBuffer => [
			:binding => binding,
			:buffer => x.storageBuffer,
			:offset => 0,
			:size => reduce(*, (x |> size)) * sizeof(eltype(x))
		],
		WGPU.GPUBuffer => [
			:binding => binding + 1,
			:buffer => x.storageBuffer,
			:offset => 0,
			:size => reduce(*, (x |> size)) * sizeof(eltype(x))
		],
	]
end


function preparePipeline(relu::ReLULayer{T}, x::AbstractArray) where T
	gpuDevice = getWgpuDevice()
	cShader = get!(task_local_storage(), (:relu, :shader, T, size(x))) do
		compileShader(relu, x)
	end
	bindingLayouts = []
	bindings = []
	append!(bindingLayouts, getBindingLayouts(relu; binding=0))
	append!(bindings, getBindings(relu, x; binding=0))
	(bindGroupLayouts, bindGroup) = WGPU.makeBindGroupAndLayout(
		gpuDevice, 
		bindingLayouts, 
		bindings
	)
	pipelineLayout = WGPU.createPipelineLayout(gpuDevice, "PipeLineLayout", bindGroupLayouts)
	computeStage = WGPU.createComputeStage(cShader.internal[], "main")
	computePipeline = WGPU.createComputePipeline(gpuDevice, "computePipeline", pipelineLayout, computeStage)
	task_local_storage((:relu, :bindingLayouts, T, size(x)), bindingLayouts)
	task_local_storage((:relu, :bindings, T, size(x)), bindings)
	task_local_storage((:relu, :layout, T, size(x)), pipelineLayout)
	task_local_storage((:relu, :pipeline, T, size(x)), computePipeline)
	task_local_storage((:relu, :bindgroup, T, size(x)), bindGroup)
	task_local_storage((:relu, :computestage, T, size(x)), computeStage)
end


function gpu_call(gpuDevice, relu::ReLULayer)
	
end


function compute(computePipeline, bindGroup, relu::ReLULayer, x)
	gpuDevice = getWgpuDevice()
	commandEncoder = WGPU.createCommandEncoder(gpuDevice, "Command Encoder")
	computePass = WGPU.beginComputePass(commandEncoder)
	WGPU.setPipeline(computePass, computePipeline)
	WGPU.setBindGroup(computePass, 0, bindGroup, UInt32[], 0, 99999)
	WGPU.dispatchWorkGroups(computePass, size(x)...)
	WGPU.endComputePass(computePass)
	WGPU.submit(gpuDevice.queue, [WGPU.finish(commandEncoder),])
end

