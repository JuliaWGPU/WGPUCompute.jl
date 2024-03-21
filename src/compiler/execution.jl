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

struct WGPUKernelObject
	kernelFunc::Function
end

mutable struct KernelBuildContext
	inargs::Dict{Symbol, Any}
	tmpargs::Array{Symbol}
	typeargs::Array{Symbol}
	stmnts::Array{Expr}
	globals::Array{Expr}
	indent::Int # Helps debugging
	kernel # TODO function body with all statements will reside here
end

function getShaderCode(f, args...; workgroupSizes=(), workgroupCount=())
	fexpr = @code_string(f(args...)) |> Meta.parse
	@capture(fexpr, function fname_(fargs__) where Targs__ fbody__ end)
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
		@const workgroupDims = Vec3{UInt32}($(UInt32.(workgroupSizes)...))
	end

	
	ins = Dict{Symbol, Any}()
	tmps = Symbol[]

	cntxt = KernelBuildContext(ins, tmps, Symbol[], Expr[], Expr[], 0, nothing)
	ins[:Targs] = Targs
	ins[:workgroupDims] = :workgroupDims

	for (inArg, symbolArg) in zip(args, fargs)
		if @capture(symbolArg, iovar_::ioType_{T_, N_})
			ins[T] = eltype(inArg)
			ins[N] = N
		end
	end
	for (idx, (inarg, symbolarg)) in enumerate(zip(args, fargs))
		if @capture(symbolarg, iovar_::ioType_{T_, N_})
		# TODO instead of assert we should branch for each case of argument
			if ioType == :WgpuArray
				dimsVar = Symbol(iovar, :Dims)
				dims = size(inarg)
			    if dims |> length < 3
		    		dims = (dims..., repeat([1,], inner=(3 - length(dims)))...)
		    	end
				push!(
					cntxt.globals,
					quote
						@const $dimsVar = Vec3{UInt32}($(UInt32.(dims)...))
					end
				)
				ins[dimsVar] = dimsVar
			end
		elseif @capture(symbolarg, iovar_::ioType_)
			if eltype(inarg) in [Float32, Int32, UInt32, Bool] # TODO we need to update this
				push!(
					cntxt.globals, 
					quote
						@const $iovar::$(eltype(inarg)) = $(Meta.parse((wgslType(inarg))))
					end
				)
				ins[iovar] = iovar
			else
				push!(
					cntxt.tmpargs,
					iovar
				)
			end
		end
	end
	
	for (idx, (inarg, symbolarg)) in enumerate(zip(args, fargs))
		if @capture(symbolarg, iovar_::ioType_{T_, N_})
		# TODO instead of assert we should branch for each case of argument
			@assert ioType == :WgpuArray #"Expecting WgpuArray Type, received $ioType instead"
			arrayLen = reduce(*, size(inarg))
			push!(
				cntxt.globals,
				quote
					@var StorageReadWrite 0 $(idx-1) $(iovar)::Array{$(eltype(inarg)), $(arrayLen)}
				end
			)
			ins[iovar] = iovar
		end
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

function wgslFunctionStatement(cntxt::KernelBuildContext, stmnt; isLast = false)
	if @capture(stmnt, a_[b_] = c_)
		stmnt = :($(wgslFunctionStatement(cntxt, a))[$(wgslFunctionStatement(cntxt, b))] = $(wgslFunctionStatement(cntxt, c)))
		push!(cntxt.stmnts, wgslAssignment(stmnt, nothing))
	elseif @capture(stmnt, a_ = b_)
		if (a in cntxt.tmpargs) && !(a in cntxt.inargs |> keys)
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
		end
		if stmnt in WGSLTypes.wgslfunctions
			return stmnt
		elseif stmnt in cntxt.tmpargs && !(stmnt in cntxt.inargs |> keys)
			return stmnt
		elseif (stmnt in cntxt.inargs |> keys)
			return :($(cntxt.inargs[stmnt]))
		else
			@error "Something is not right with $stmnt expr"
		end
	elseif typeof(stmnt) <: Number
		return stmnt
	elseif @capture(stmnt, a_[b_])
		asub = wgslFunctionStatement(cntxt, a)
		bsub = wgslFunctionStatement(cntxt, b)
		return :($asub[$bsub])
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
		elseif f in cntxt.inargs[:Targs]
			return cntxt.inargs[f]
		else
			return :($f($x, $y))
		end
	elseif @capture(stmnt, f_(x__))
		(fcall, xargs...) = map((xArg) -> wgslFunctionStatement(cntxt, xArg), [f, x...])
		return :($fcall($(xargs...)))
	elseif @capture(stmnt, return t_)
		push!(cntxt.stmnts, (wgslType(t)))
	elseif @capture(stmnt, if cond_ ifblock__ end)
		if cond == true
			wgslFunctionStatements(cntxt, ifblock)
		end
	elseif @capture(stmnt, if cond_ ifBlock__ else elseBlock__ end)
		if eval(cond) == true
			wgslFunctionStatements(cntxt, ifBlock)
		else
			wgslFunctionStatements(cntxt, elseBlock)
		end
	elseif @capture(stmnt, for idx_ in range_ block__ end)
		newcntxt = KernelBuildContext(Dict{Symbol, Any}(), Symbol[], Symbol[], Expr[], Expr[], 0, nothing)
		code = quote end
		#push!(code.args, :(for idx_ in range)
		push!(cntxt.tmpargs, idx)
		for loopstmnt in block
			wgslFunctionStatement(newcntxt, loopstmnt)
		end
		newblock = newcntxt.stmnts
		push!(cntxt.stmnts, Expr(:for, Expr(:(=), idx, range), quote $(newblock...) end))
		@infiltrate
	else
		@error "Failed to capture statment : $stmnt !!"
	end
end

function compileShader(f, args...; workgroupSizes=(), workgroupCount=())
	shaderSrc = getShaderCode(f, args...; workgroupSizes=workgroupSizes, workgroupCount=workgroupCount)
	@info shaderSrc |> MacroTools.striplines |> MacroTools.flatten
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

function preparePipeline(f::Function, args...; workgroupSizes=(), workgroupCount=())
	gpuDevice = WGPUCompute.getWgpuDevice()
	cShader = get!(task_local_storage(), (f, :shader, eltype.(args), size.(args))) do
		compileShader(f, args...; workgroupSizes=workgroupSizes, workgroupCount=workgroupCount)
	end
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
	# task_local_storage((nameof(f), :bindgrouplayout, T, size(x)), bindGroupLayouts)
	task_local_storage((nameof(f), :bindings, eltype.(args), size.(args)), bindings)
	task_local_storage((nameof(f), :bindinglayouts, eltype.(args), size.(args)), bindingLayouts)
	task_local_storage((nameof(f), :layout, eltype.(args), size.(args)), pipelineLayout)
	task_local_storage((nameof(f), :pipeline, eltype.(args), size.(args)), computePipeline)
	task_local_storage((nameof(f), :bindgroup, eltype.(args), size.(args)), pipelineLayout.bindGroup)
	task_local_storage((nameof(f), :computestage, eltype.(args), size.(args)), computeStage)
end

function compute(f, args...; workgroupSizes=(), workgroupCount=())
	gpuDevice = WGPUCompute.getWgpuDevice()
	commandEncoder = WGPUCore.createCommandEncoder(gpuDevice, "Command Encoder")
	computePass = WGPUCore.beginComputePass(commandEncoder)
	WGPUCore.setPipeline(computePass, task_local_storage((nameof(f), :pipeline, eltype.(args), size.(args))))
	WGPUCore.setBindGroup(computePass, 0, task_local_storage((nameof(f), :bindgroup, eltype.(args), size.(args))), UInt32[], 0, 99999)
	WGPUCore.dispatchWorkGroups(computePass, workgroupCount...) # workgroup size needs work here
	WGPUCore.endComputePass(computePass)
	WGPUCore.submit(gpuDevice.queue, [WGPUCore.finish(commandEncoder),])
end

function kernelFunc(funcExpr)
	if 	@capture(funcExpr, function fname_(fargs__) where Targs__ fbody__ end)
		kernelfunc = quote
			function $fname(args::Tuple{WgpuArray}, workgroupSizes, workgroupCount)
				$preparePipeline($(funcExpr), args...)
				$compute($(funcExpr), args...; workgroupSizes=workgroupSizes, workgroupCount=workgroupCount)
				return nothing
			end
		end |> unblock
		return esc(quote $kernelfunc end)
	else
		error("Couldnt capture function")
	end
end

function getFunctionBlock(func, args)
	fString = CodeTracking.definition(String, which(func, args))
	return Meta.parse(fString |> first)
end

function wgpuCall(kernelObj::WGPUKernelObject, args...)
	kernelObj.kernelFunc(args...)
end

macro wgpukernel(launch, wgSize, wgCount, ex)
	code = quote end
	@gensym f_var kernel_f kernel_args kernel_tt kernel
	if @capture(ex, fname_(fargs__))
		(vars, var_exprs) = assign_args!(code, fargs)
		push!(code.args, quote
				$kernel_args = ($(var_exprs...),)
				$kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
				kernel = function wgpuKernel(args...)
					$preparePipeline($fname, args...; workgroupSizes=$wgSize, workgroupCount=$wgCount)
					$compute($fname, args...; workgroupSizes=$wgSize, workgroupCount=$wgCount)
				end
				if $launch == true
					wgpuCall(WGPUKernelObject(kernel), $(kernel_args)...)
				else
					WGPUKernelObject(kernel)
				end
			end
		)
	end
	return esc(code)
end
