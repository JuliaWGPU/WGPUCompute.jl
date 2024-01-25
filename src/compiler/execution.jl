export @wgpukernel, getShaderCode, emitWGSLJuliaBody

using MacroTools
using CodeTracking
using Lazy

# TODO remove
using WGPUCompute
using Infiltrator
using MLStyle
using MLStyle: @match
using JuliaVariables

function WGSLTypes.wgslType(a::JuliaVariables.Var)
	if a.is_mutable && !(a.is_global)
		if isdefined(Base, a.name)
			return a.name
		else
			return :(@var $(a.name))
		end
	elseif a.is_global
		return a.name
	elseif !(a.is_global)
		return a.name
	else
		@error "This JuliaVariable is not captured"
	end
end

# copied from https://github.com/JuliaStaging/JuliaVariables.jl
unwrap_scoped(ex) =
	@match ex begin
	    Expr(:scoped, _, a) => unwrap_scoped(a)
	    Expr(head, args...) => Expr(head, map(unwrap_scoped, args)...)
	    a => a
	end

resolve! = unwrap_scoped ∘ MacroTools.striplines ∘ solve! ∘ simplify_ex

function getShaderCode(f, args::WgpuArray{T, N}...) where {T, N}
	fexpr = @code_string(f(args...)) |> Meta.parse
	@capture(fexpr, @wgpukernel workgroupSizes_ workgroupCount_ function fname_(fargs__) where Targs__ fbody__ end)

	workgroupSizes = Meta.eval(workgroupSizes)
	workgroupCount = Meta.eval(workgroupCount)
	originArgs = fargs[:]
	
	builtinArgs = [
		:(@builtin(global_invocation_id, global_id::Vec3{UInt32})),
		:(@builtin(local_invocation_id, local_id::Vec3{UInt32})),
		:(@builtin(num_workgroups, num_workgroups::Vec3{UInt32})),
		:(@builtin(workgroup_id, workgroup_id::Vec3{UInt32})),
	]
	
    # TODO used `repeat` since `ones` is causing issues.
    # interesting bug to raise.
    if workgroupSizes |> length < 3
    	workgroupSizes = (workgroupSizes..., repeat([1,], inner=(3 - length(workgroupSizes)))...)
    end
  
	code = quote
		@const workgroupDims = Vec3{Int32}($(workgroupSizes...))
		struct IOArray
			data::WArray{$T}
		end
	end

	for (idx, arg) in enumerate(fargs)
		if @capture(arg, a_::b_)
			push!(
				code.args,
				quote
					@var StorageReadWrite 0 $(idx-1) $a::IOArray
				end |> unblock
			)
		else
			@error "Could not capture input arguments"
		end
	end

	fquote = quote
		function $(fname)($(builtinArgs...))
			$(fbody...)
		end
	end |> resolve!

    push!(code.args,
		:(@compute @workgroupSize($(workgroupSizes...)) $(fquote.args...))
   	)
	
    return (code |> MacroTools.striplines)
end

function compileShader(f, args::WgpuArray{T, N}...) where {T, N}
	shaderSrc = getShaderCode(f, args...)
	cShader = nothing
	try
		cShader = createShaderObj(WGPUCompute.getWgpuDevice(), shaderSrc; savefile=true)
	catch(e)
		@info e
		rethrow(e)
	end
	@info cShader.src
	task_local_storage((f, :shader, T, N, size.(args)), cShader)
	return cShader
end

function preparePipeline(f, args::WgpuArray{T, N}...) where {T, N}
	gpuDevice = WGPUCompute.getWgpuDevice()
	cShader = get!(task_local_storage(), (f, :shader, T, size.(args))) do
		compileShader(f, args...)
	end
	bindingLayouts = []
	bindings = []
	
	for (binding, arg) in enumerate(args)
		push!(bindingLayouts, 
			WGPUCore.WGPUBufferEntry => [
				:binding => binding - 1,
				:visibility => "Compute",
				:type => "Storage"
			],
		)
	end

	for (binding, arg) in enumerate(args)
		push!(bindings, 
			WGPUCore.GPUBuffer => [
				:binding => binding - 1,
				:buffer => arg.storageBuffer,
				:offset => 0,
				:size => reduce(*, (arg |> size)) * sizeof(eltype(arg))
			],
		)
	end

	pipelineLayout = WGPUCore.createPipelineLayout(gpuDevice, "PipeLineLayout", bindingLayouts, bindings)
	computeStage = WGPUCore.createComputeStage(cShader.internal[], f |> string)
	computePipeline = WGPUCore.createComputePipeline(gpuDevice, "computePipeline", pipelineLayout, computeStage)
	# task_local_storage((nameof(f), :bindgrouplayout, T, size(x)), bindGroupLayouts)
	task_local_storage((nameof(f), :bindings, T, size.(args)), bindings)
	task_local_storage((nameof(f), :bindinglayouts, T, size.(args)), bindingLayouts)
	task_local_storage((nameof(f), :layout, T, size.(args)), pipelineLayout)
	task_local_storage((nameof(f), :pipeline, T, size.(args)), computePipeline)
	task_local_storage((nameof(f), :bindgroup, T, size.(args)), pipelineLayout.bindGroup)
	task_local_storage((nameof(f), :computestage, T, size.(args)), computeStage)
end

function compute(f, args::WgpuArray{T, N}...; workgroupSizes=(), workgroupCount=()) where {T, N}
	gpuDevice = WGPUCompute.getWgpuDevice()
	commandEncoder = WGPUCore.createCommandEncoder(gpuDevice, "Command Encoder")
	computePass = WGPUCore.beginComputePass(commandEncoder)
	WGPUCore.setPipeline(computePass, task_local_storage((nameof(f), :pipeline, T, size.(args))))
	WGPUCore.setBindGroup(computePass, 0, task_local_storage((nameof(f), :bindgroup, T, size.(args))), UInt32[], 0, 99999)
	WGPUCore.dispatchWorkGroups(computePass, workgroupCount...) # workgroup size needs work here
	WGPUCore.endComputePass(computePass)
	WGPUCore.submit(gpuDevice.queue, [WGPUCore.finish(commandEncoder),])
end

function kernelFunc(funcExpr; workgroupSizes=nothing, workgroupCount=nothing)
	workgroupSizes = Meta.eval(workgroupSizes)
	workgroupCount = Meta.eval(workgroupCount)
	if 	@capture(funcExpr, function fname_(fargs__) where Targs__ fbody__ end)
		kernelfunc = quote
			function $fname(args::WgpuArray{T, N}...) where {T, N}
				$preparePipeline($(funcExpr), args...)
				$compute($(funcExpr), args...; workgroupSizes=$workgroupSizes, workgroupCount=$workgroupCount)
				return nothing
			end
		end
		return esc(kernelfunc)
	else
		error("Couldnt capture function")
	end
end

macro wgpukernel(workgroupSizes, workgroupCount, expr)
	kernelFunc(expr; workgroupSizes=workgroupSizes, workgroupCount=workgroupCount)
end
