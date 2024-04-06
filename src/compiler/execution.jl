using WGPUTranspiler
using WGPUTranspiler: TypeExpr, Scope, WorkGroupDims, computeBlock, ComputeBlock
using Tracy
using Tracy: @tracepoint
export WGPUKernelContext

# getSize is temporary solution for circumvent for function in kernel arguments
# could be common solution non array arguments ...
getSize(a::WgpuArray) = size(a)
getSize(a::Function) = ()
getSize(a::Number) = ()
getSize(::typeof(+)) = ()
getSize(::typeof(-)) = ()
getSize(::typeof(*)) = ()
getSize(::typeof(/)) = ()

# Small hack to support TypeExpr of WGPUTranspiler. 
# TODO think of better abstraction. 
function WGPUTranspiler.typeInfer(scope::Scope, tExpr::TypeExpr, v::Val{:WgpuArray})
	return WgpuArray{map(x -> WGPUTranspiler.typeInfer(scope, x), tExpr.types)...}
end

export @wgpukernel, getShaderCode, WGPUKernelObject, wgpuCall

function getWgpuDevice()
	get!(task_local_storage(), :WGPUDevice) do
		WGPUCore.getDefaultDevice(nothing)
	end
end

using MacroTools
using CodeTracking
using Lazy
using Infiltrator

# TODO remove
using WGPUCompute
using Infiltrator

struct WGPUKernelContext end

struct WGPUDeviceKernel
	kernelFunc::Function
end

function getShaderCode(f, args; workgroupSize=(), workgroupCount=(), shmem=())
	fexpr = @code_string(f(args...)) |> Meta.parse |> MacroTools.striplines
	scope = Scope(Dict(), Dict(), Dict(), 0, nothing, quote end)
	cblk = computeBlock(scope, true, workgroupSize, workgroupCount, shmem, f, args, fexpr)
	tblk = transpile(scope, cblk)
    return tblk
end

function compileShader(f, args; workgroupSize=(), workgroupCount=(), shmem=())
	shaderSrc = getShaderCode(f, args; workgroupSize=workgroupSize, workgroupCount=workgroupCount, shmem=shmem)
	cShader = nothing
	try
		cShader = createShaderObj(WGPUCompute.getWgpuDevice(), shaderSrc; savefile=true)
	catch(e)
		@info e
		rethrow(e)
	end
	task_local_storage((f, :shader, eltype.(args), getSize.(args), workgroupSize, workgroupCount, shmem), cShader)
	return cShader
end


function compute(f::Function, args; workgroupSize=(), workgroupCount=(), shmem=())
	cShader = get!(task_local_storage(), 
			(f, :shader, eltype.(args), getSize.(args), workgroupSize, workgroupCount, shmem)) do
				compileShader(f, args; 
					workgroupSize=workgroupSize, 
					workgroupCount=workgroupCount, 
					shmem=shmem
				)
			end
	gpuDevice = WGPUCompute.getWgpuDevice()
	bindingLayouts = []
	bindings = []

	bindingCount = 0
	for (_, arg) in enumerate(args)
		if typeof(arg) <: WgpuArray
			bindingCount += 1
			push!(bindingLayouts, 
				WGPUCore.WGPUBufferEntry => [
					:binding => bindingCount - 1,
					:visibility => "Compute",
					:type => "Storage"
				],
			)
		end
	end

	bindingCount = 0

	for (_, arg) in enumerate(args)
		if typeof(arg) <: WgpuArray
			bindingCount += 1
			push!(bindings, 
				WGPUCore.GPUBuffer => [
					:binding => bindingCount - 1,
					:buffer => arg.storageBuffer,
					:offset => 0,
					:size => reduce(*, (arg |> size)) * sizeof(eltype(arg))
				],
			)
		end
	end

	pipelineLayout = WGPUCore.createPipelineLayout(gpuDevice, "PipeLineLayout", bindingLayouts, bindings)
	computeStage = WGPUCore.createComputeStage(cShader.internal[], f |> string)
	computePipeline = WGPUCore.createComputePipeline(gpuDevice, "computePipeline", pipelineLayout, computeStage)
	commandEncoder = WGPUCore.createCommandEncoder(gpuDevice, "Command Encoder")
	computePass = WGPUCore.beginComputePass(commandEncoder)
	WGPUCore.setPipeline(computePass, computePipeline)
	WGPUCore.setBindGroup(computePass, 0, pipelineLayout.bindGroup, UInt32[], 0, 99999)
	WGPUCore.dispatchWorkGroups(computePass, workgroupCount...) # workgroup size needs work here
	WGPUCore.endComputePass(computePass)
	WGPUCore.submit(gpuDevice.queue, [WGPUCore.finish(commandEncoder),])
end


function getFunctionBlock(func, args)
	fString = CodeTracking.definition(String, which(func, args))
	return Meta.parse(fString |> first)
end

function deviceKernel(fname, fargs; argTypes, wgSize, wgCount, shmem)
	kernelInstance = get!(task_local_storage(), (fname, argTypes, getSize.(fargs), wgSize, wgCount, shmem)) do
		WGPUDeviceKernel(
			function wgpuKernel(args)
				# TODO submit or not ? @async ...
				compute(fname, args; workgroupSize=wgSize, workgroupCount=wgCount, shmem=shmem)
			end
		)
	end
	return kernelInstance.kernelFunc
end

macro wgpukernel(launch, wgSize, wgCount, shmem, ex)
	code = quote end
	@gensym f_var kernel_f kernel_args kernel_tt kernel
	if @capture(ex, fname_(fargs__))
		(vars, var_exprs) = assign_args!(code, fargs)
		push!(
			code.args,
			quote
				$kernel_args = ($(var_exprs...),)
				$kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
				$kernel = $deviceKernel(
						$fname, 
						$kernel_args;
						argTypes=$kernel_tt, 
						wgSize=$wgSize, 
						wgCount=$wgCount, 
						shmem=$shmem,
					)
				if $launch == true
					$kernel($(kernel_args))
				else
					$kernel
				end
			end
		)
	end
	# TODO
	# THIS IS STALE until Kernel abstractions (KA) implementation
	# elseif @capture(ex, function fname_(fargs__) where Targs__ fbody__ end)
	#	push!(
	#		code.args, 
	#		quote
	#			kernel = function wgpuKernel(args...)
	#				$preparePipeline($ex, args...; workgroupSize=$wgSize, workgroupCount=$wgCount)
	#				$compute($ex, args...; workgroupSize=$wgSize, workgroupCount=$wgCount)
	#			end
	#			WGPUKernelContext()
	#		end
	#	)
	return esc(code)
end
