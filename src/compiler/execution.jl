export @wgpukernel, getShaderCode, emitWGSLJuliaBody

using MacroTools
using CodeTracking
using Lazy

# TODO remove
using WGPUCompute
using Infiltrator

mutable struct KernelContext
	inargs::Dict{Symbol, Any}
	outargs::Dict{Symbol, Any}
	tmpargs::Array{Symbol}
	typeargs::Array{Symbol}
	stmnts::Array{Expr}
	globals::Array{Expr}
	indent::Int # Helps debugging
	kernel # TODO function body with all statements will reside here
end

function getShaderCode(f, args::WgpuArray...)
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
	end
	
	ins = Dict{Symbol, Any}()
	outs = Dict{Symbol, Any}()

	cntxt = KernelContext(ins, outs, Symbol[], Symbol[], Expr[], Expr[], 0, nothing)
	
	for (idx, (inarg, symbolarg)) in enumerate(zip(args, fargs))
		@capture(symbolarg, iovar_::ioType_{T_, N_})
		# TODO instead of assert we should branch for each case of argument
		@assert ioType == :WgpuArray "Expecting WgpuArray Type, received $ioType instead"
		arrayLen = reduce(*, size(inarg))
		push!(
			cntxt.globals,
			quote
				@var StorageReadWrite 0 $(idx-1) $(iovar)::Array{$(eltype(inarg)), $(arrayLen)}
			end
		)
		ins[iovar] = iovar
	end
				
	wgslFunctionStatements(cntxt, fbody)
	
	fquote = quote
		function $(fname)($(builtinArgs...))
			$((cntxt.stmnts)...)
		end
	end
	
	push!(code.args, cntxt.globals...)
	
    push!(code.args,
		:(@compute @workgroupSize($(workgroupSizes...)) $(fquote.args...))
   	)
	
    return code
end

function wgslAssignment(expr::Expr, prefix::Union{Nothing, Symbol})
	@capture(expr, a_ = b_) || error("Expecting simple assignment a = b")
	return ifelse(prefix==:let, :(@let $a = $b), :($a = $b))
end


function wgslFunctionStatements(cntxt, stmnts)
	for (i, stmnt) in enumerate(stmnts)
		wgslFunctionStatement(cntxt, stmnt; isLast=(length(stmnts) == i))
	end
end

function wgslFunctionStatement(cntxt::KernelContext, stmnt; isLast = false)
	if @capture(stmnt, a_[b_] = c_)
		stmnt = :($(wgslFunctionStatement(cntxt, a))[$(wgslFunctionStatement(cntxt, b))] = $(wgslFunctionStatement(cntxt, c)))
		push!(cntxt.stmnts, wgslAssignment(stmnt, nothing))
	elseif @capture(stmnt, a_ = b_)
		if (a in cntxt.tmpargs) && !(a in cntxt.inargs |> keys) && !(a in cntxt.outargs |> keys)
			stmnt = :($(wgslFunctionStatement(cntxt, a)) = $(wgslFunctionStatement(cntxt, b)))
			push!(cntxt.stmnts, wgslAssignment(stmnt, nothing))
		else
			push!(cntxt.tmpargs, a)
			stmnt = :($(wgslFunctionStatement(cntxt, a)) = $(wgslFunctionStatement(cntxt, b)))
			push!(cntxt.stmnts, wgslAssignment(stmnt, :let))
		end
	elseif @capture(stmnt, a_.b_)
		return :($(wgslFunctionStatement(cntxt, a)).$b)
	elseif typeof(stmnt) == Symbol
		if stmnt == :globalId # TODO 
			return :global_id
		elseif stmnt == :dispatchDims # TODO 
			return :num_workgroups
		elseif stmnt == :localId
			return :local_id
		elseif stmnt == :workgroupId
			return :workgroup_id
		elseif stmnt == :workgroupDims
			return :num_workgroups
		end
		if stmnt in cntxt.tmpargs && !(stmnt in cntxt.inargs |> keys) && !(stmnt in cntxt.outargs |> keys)
			return stmnt
		elseif (stmnt in cntxt.inargs |> keys)
			return :($(cntxt.inargs[stmnt]))
		elseif (stmnt in cntxt.outargs |> keys)
			iovar = Symbol(:output, length(cntxt.outargs |> keys) + 1)
			return :($(cntxt.outargs[stmnt]))
		else
			@error "Something is not right with $stmnt expr"
		end
	elseif typeof(stmnt) <: Number
		return stmnt
	elseif @capture(stmnt, a_[b_])
		asub = wgslFunctionStatement(cntxt, a)
		bsub = wgslFunctionStatement(cntxt, b)
		return :($asub[$bsub])
	elseif @capture(stmnt, @let t_ | @let t__)
		push!(cntxt.stmnts, stmnt)
	elseif @capture(stmnt, @var t_ | @let t__)
		push!(cntxt.stmnts, stmnt)
	elseif @capture(stmnt, a_ += b_)
		wgslFunctionStatement(cntxt, :($a = $(wgslFunctionStatement(cntxt, a)) + $(wgslFunctionStatement(cntxt, b))))
	elseif @capture(stmnt, a_ -= b_)
		wgslFunctionStatement(cntxt, :($a = $(wgslFunctionStatement(cntxt, a)) - $(wgslFunctionStatement(cntxt, b))))
	elseif @capture(stmnt, a_ *= b_)
		wgslFunctionStatement(cntxt, :($a = $(wgslFunctionStatement(cntxt, a)) * $(wgslFunctionStatement(cntxt, b))))
	elseif @capture(stmnt, a_ /= b_)
		wgslFunctionStatement(cntxt, :($a = $(wgslFunctionStatement(cntxt, a)) / $(wgslFunctionStatement(cntxt, b))))
	elseif @capture(stmnt, f_(x_, y_))
		if f in (:*, :-, :+, :/)
			x = wgslFunctionStatement(cntxt, x)
			y = wgslFunctionStatement(cntxt, y)
			return :($f($x, $y))
		else
			return :($f($x, $y))
		end
	elseif @capture(stmnt, f_(x__))
		x = tuple(x...)
		return :($f($(x...)))
	elseif @capture(stmnt, return t_)
		push!(cntxt.stmnts, (wgslType(t)))
	elseif @capture(stmnt, if cond_ ifblock__ end)
		if cond == true
			wgslFunctionStatements(io, ifblock)
		end
	elseif @capture(stmnt, if cond_ ifBlock__ else elseBlock__ end)
		if eval(cond) == true
			wgslFunctionStatements(io, ifBlock)
		else
			wgslFunctionStatements(io, elseBlock)
		end
	else
		@error "Failed to capture statment : $stmnt !!"
	end
end

function compileShader(f, args::WgpuArray...)
	shaderSrc = getShaderCode(f, args...)
	cShader = nothing
	try
		cShader = createShaderObj(WGPUCompute.getWgpuDevice(), shaderSrc; savefile=true)
	catch(e)
		@info e
		rethrow(e)
	end
	@info cShader.src
	task_local_storage((f, :shader, eltype.(args), size.(args)), cShader)
	return cShader
end

function preparePipeline(f, args::WgpuArray...)
	gpuDevice = WGPUCompute.getWgpuDevice()
	cShader = get!(task_local_storage(), (f, :shader, eltype.(args), size.(args))) do
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
	task_local_storage((nameof(f), :bindings, eltype.(args), size.(args)), bindings)
	task_local_storage((nameof(f), :bindinglayouts, eltype.(args), size.(args)), bindingLayouts)
	task_local_storage((nameof(f), :layout, eltype.(args), size.(args)), pipelineLayout)
	task_local_storage((nameof(f), :pipeline, eltype.(args), size.(args)), computePipeline)
	task_local_storage((nameof(f), :bindgroup, eltype.(args), size.(args)), pipelineLayout.bindGroup)
	task_local_storage((nameof(f), :computestage, eltype.(args), size.(args)), computeStage)
end

function compute(f, args::WgpuArray...; workgroupSizes=(), workgroupCount=())
	gpuDevice = WGPUCompute.getWgpuDevice()
	commandEncoder = WGPUCore.createCommandEncoder(gpuDevice, "Command Encoder")
	computePass = WGPUCore.beginComputePass(commandEncoder)
	WGPUCore.setPipeline(computePass, task_local_storage((nameof(f), :pipeline, eltype.(args), size.(args))))
	WGPUCore.setBindGroup(computePass, 0, task_local_storage((nameof(f), :bindgroup, eltype.(args), size.(args))), UInt32[], 0, 99999)
	WGPUCore.dispatchWorkGroups(computePass, workgroupCount...) # workgroup size needs work here
	WGPUCore.endComputePass(computePass)
	WGPUCore.submit(gpuDevice.queue, [WGPUCore.finish(commandEncoder),])
end

function kernelFunc(funcExpr; workgroupSizes=nothing, workgroupCount=nothing)
	workgroupSizes = Meta.eval(workgroupSizes)
	workgroupCount = Meta.eval(workgroupCount)
	if 	@capture(funcExpr, function fname_(fargs__) where Targs__ fbody__ end)
		kernelfunc = quote
			function $fname(args::WgpuArray...)
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
