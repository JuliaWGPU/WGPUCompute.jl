using WGSLTypes
using WGPUCore
using LinearAlgebra
using StaticArrays

abstract type AbstractLayer{T} end
abstract type ActivationLayer{T} <: AbstractLayer{T} end

# list of elementwise functions
elementFuncs = [
	"ReLU",
	"ELU",
]

struct ReLULayer{T} <: ActivationLayer{T}
end

# TODO compileShader only needs size of x
function compileShader(relu::ReLULayer{T}, x::WgpuArray{T}) where T
	shaderSrc = getShaderCode(relu, x)
	cShader = nothing
	try
		cShader = createShaderObj(getWgpuDevice(), shaderSrc; savefile = true)
	catch(e)
		@info e
		rethrow(e)
	end
	@info cShader.src
	task_local_storage((:relu, :shader, T, size(x)), cShader)
	return cShader
end


function (relu::ReLULayer{T})(x::WgpuArray{T}) where T
	dims = size(x)
	# get!(task_local_storage(), (:relu, T, size(x))) do
		# preparePipeline(relu, x)
	# end
	y = similar(x)
	preparePipeline(relu, x, y)
	compute(relu, x)
	return y
end


function getShaderCode(activation::ReLULayer{T}, x::WgpuArray{T}) where T
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
		
		@compute @workgroupSize(8, 8, 4) function main(@builtin global_invocation_id => global_id::Vec3{UInt32})
			@let gIdx = global_id.x*global_id.y + global_id.z;
			@let value = input0.data[gIdx]
			output0.data[gIdx] = max(value, $(zero(eltype(activation))))
		end
	end
end


function getBindingLayouts(relu::ReLULayer; binding=0)
	bindingLayouts = [
		WGPUCore.WGPUBufferEntry => [
			:binding => binding,
			:visibility => "Compute",
			:type => "Storage"
		],
		WGPUCore.WGPUBufferEntry => [
			:binding => binding + 1,
			:visibility => "Compute",
			:type => "Storage"
		]
	]
end


function getBindings(relu::ReLULayer, x, y; binding=0)
	bindings = [
		WGPUCore.GPUBuffer => [
			:binding => binding,
			:buffer => x.storageBuffer,
			:offset => 0,
			:size => reduce(*, (x |> size)) * sizeof(eltype(x))
		],
		WGPUCore.GPUBuffer => [
			:binding => binding + 1,
			:buffer => y.storageBuffer,
			:offset => 0,
			:size => reduce(*, (y |> size)) * sizeof(eltype(x))
		],
	]
end


function preparePipeline(relu::ReLULayer{T}, x::WgpuArray{T}, y::WgpuArray{T}) where T
	gpuDevice = getWgpuDevice()
	cShader = get!(task_local_storage(), (:relu, :shader, T, size(x))) do
		compileShader(relu, x)
	end
	bindingLayouts = []
	bindings = []
	append!(bindingLayouts, getBindingLayouts(relu; binding=0))
	append!(bindings, getBindings(relu, x, y; binding=0))
	pipelineLayout = WGPUCore.createPipelineLayout(gpuDevice, "PipeLineLayout", bindingLayouts, bindings)
	computeStage = WGPUCore.createComputeStage(cShader.internal[], "main")
	computePipeline = WGPUCore.createComputePipeline(gpuDevice, "computePipeline", pipelineLayout, computeStage)
	# task_local_storage((:relu, :bindgrouplayout, T, size(x)), pipelineLayout.bindGroupLayouts)
	task_local_storage((:relu, :bindings, T, size(x)), bindings)
	task_local_storage((:relu, :bindinglayouts, T, size(x)), bindingLayouts)
	task_local_storage((:relu, :layout, T, size(x)), pipelineLayout)
	task_local_storage((:relu, :pipeline, T, size(x)), computePipeline)
	task_local_storage((:relu, :bindgroup, T, size(x)), pipelineLayout.bindGroup)
	task_local_storage((:relu, :computestage, T, size(x)), computeStage)
end


function gpu_call(gpuDevice, relu::ReLULayer)
	
end


function compute(relu::ReLULayer{T}, x::WgpuArray{T}) where T
	gpuDevice = getWgpuDevice()
	commandEncoder = WGPUCore.createCommandEncoder(gpuDevice, "Command Encoder")
	computePass = WGPUCore.beginComputePass(commandEncoder)
	WGPUCore.setPipeline(computePass, task_local_storage((:relu, :pipeline, T, size(x))))
	WGPUCore.setBindGroup(computePass, 0, task_local_storage((:relu, :bindgroup, T, size(x))), UInt32[], 0, 99999)
	WGPUCore.dispatchWorkGroups(computePass, size(x)...)
	WGPUCore.endComputePass(computePass)
	WGPUCore.submit(gpuDevice.queue, [WGPUCore.finish(commandEncoder),])
end


