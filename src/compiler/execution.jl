export @kernel, getShaderCode, emitWGSLJuliaBody

"""
@wgpu [kwargs...] func(args...)

High-level interface for executing code on a GPU. The `@wgpu` macro should prefix a call,
with `func` a callable function or object that should return nothing. It will be compiled to
a Metal function upon first use, and to a certain extent arguments will be converted and
managed automatically using `wgpuconvert`. Finally, a call to `wgpucall` is
performed, creating a command buffer in the current global command queue then committing it.

There is one supported keyword argument that influences the behavior of `@wgpu`.
- `launch`: whether to launch this kernel, defaults to `true`. If `false` the returned
  kernel object should be launched by calling it and passing arguments again.
"""

using MacroTools
using CodeTracking
using Lazy

# TODO remove
using WGPUCompute

function getShaderCode(f, x)
	T = eltype(x)
	fexpr = @code_string(f(x)) |> Meta.parse
	@capture(fexpr, @kernel function fname_(fargs__) where Targs__ fbody__ end)
	
	originArgs = fargs[:]
	builtinArgs = [:(@builtin global_invocation_id => global_id::Vec3{UInt32})]
	
	"""
	
	# This section needs to be generated too
	# Since kernels can have more than one input
	# we need to accomodate for that.
	
    code = quote
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
    end
    
    """
	
	code = quote
		struct IOArray
			data::WArray{$T}
		end
	end
	
	cntxt = emitWGSLJuliaBody(fbody, fargs)
	
	fquote = quote
		function $(fname)($(builtinArgs...))
			$((cntxt.stmnts)...)
		end
	end
	
	push!(code.args, cntxt.globals...)
	
    push!(code.args,
		:(@compute @workgroupSize(8, 8, 4) $(fquote.args...))
   	)
	
    return code
end

macro tt(func)
	@show func
	fexpr = @code_expr(call(func))
end

macro wgpukernel(workgroupsizeExpr, dispatchExpr, func)
	@capture(func, f_(x_))
	wgpu(workgroupsizeExpr, dispatchExpr, func)
end

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

# @forward KernelContext.args push!
function emitWGSLJuliaBody(fbody, inargs)
	ins = Dict{Symbol, Any}()
	outs = Dict{Symbol, Any}()
	
	cntxt = KernelContext(ins, outs, Symbol[], Symbol[], Expr[], Expr[], 0, nothing)
	
	for (idx, arg) in enumerate(inargs)
		if @capture(arg, a_::b_)
			iovar = Symbol(:input, idx-1)
			push!(cntxt.globals, quote
				@var StorageReadWrite 0 $(idx-1) $(iovar)::IOArray
			end)
			ins[a] = iovar
		else
			@error "Could not capture input arguments"
		end
	end
	
	# TODO this is stupid but good first implementation maybe
	# This assumes that the output argument is lhs of last stmnt
	if @capture(fbody[end], a_[b_] = c_) || @capture(fbody[end], a_=b_)
		idx = length(ins)
		iovar = Symbol(:ouput, idx)
		outs[a] = iovar
		push!(cntxt.globals, quote
			@var StorageReadWrite 0 $(idx) $(iovar)::IOArray
		end)
	elseif false
		# TODO captures others like return statements
		# TODO or just symbol
	end
	
	wgslFunctionStatements(cntxt, fbody)
	cntxt
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
		elseif stmnt == :localId
			return :local_id
		end
		if stmnt in cntxt.tmpargs && !(stmnt in cntxt.inargs |> keys) && !(stmnt in cntxt.outargs |> keys)
			return stmnt
		elseif (stmnt in cntxt.inargs |> keys)
			return :($(cntxt.inargs[stmnt]).data)
		elseif (stmnt in cntxt.outargs |> keys)
			iovar = Symbol(:output, length(cntxt.outargs |> keys) + 1)
			return :($(cntxt.outargs[stmnt]).data)
		else
			@error "Something is not right with $stmnt expr"
		end
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
		return :($f(($x...)))
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

function compileShader(f, x::WgpuArray{T, N}) where {T, N}
	shaderSrc = getShaderCode(f, x)
	cShader = nothing
	try
		cShader = createShaderObj(WGPUCompute.getWgpuDevice(), shaderSrc; savefile=true)
	catch(e)
		@info e
		rethrow(e)
	end
	@info cShader.src
	task_local_storage((f, :shader, T, N, size(x)), cShader)
	return cShader
end

function preparePipeline(f, x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	gpuDevice = WGPUCompute.getWgpuDevice()
	cShader = get!(task_local_storage(), (f, :shader, T, size(x))) do
		compileShader(f,  x)
	end
	bindingLayouts = []
	bindings = []
	append!(bindingLayouts, getBindingLayouts(Val(nameof(f)); binding=0))
	append!(bindings, getBindings(Val(nameof(f)), x, y; binding=0))
	(bindGroupLayouts, bindGroup) = WGPUCore.makeBindGroupAndLayout(
		gpuDevice,
		bindingLayouts,
		bindings
	)
	pipelineLayout = WGPUCore.createPipelineLayout(gpuDevice, "PipeLineLayout", bindGroupLayouts)
	computeStage = WGPUCore.createComputeStage(cShader.internal[], f |> string)
	computePipeline = WGPUCore.createComputePipeline(gpuDevice, "computePipeline", pipelineLayout, computeStage)
	task_local_storage((nameof(f), :bindgrouplayout, T, size(x)), bindGroupLayouts)
	task_local_storage((nameof(f), :bindings, T, size(x)), bindings)
	task_local_storage((nameof(f), :bindinglayouts, T, size(x)), bindingLayouts)
	task_local_storage((nameof(f), :layout, T, size(x)), pipelineLayout)
	task_local_storage((nameof(f), :pipeline, T, size(x)), computePipeline)
	task_local_storage((nameof(f), :bindgroup, T, size(x)), bindGroup)
	task_local_storage((nameof(f), :computestage, T, size(x)), computeStage)
end

# TODO this function has to be generated based on
# input and output buffers
function getBindingLayouts(::Val{:Relu}; binding=0)
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

# TODO this function has to be generated based on
# input and output buffers
function getBindings(::Val{:Relu}, x, y; binding=0)
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

function compute(f, x::WgpuArray{T, N}) where {T, N}
	gpuDevice = WGPUCompute.getWgpuDevice()
	commandEncoder = WGPUCore.createCommandEncoder(gpuDevice, "Command Encoder")
	computePass = WGPUCore.beginComputePass(commandEncoder)
	WGPUCore.setPipeline(computePass, task_local_storage((nameof(f), :pipeline, T, size(x))))
	WGPUCore.setBindGroup(computePass, 0, task_local_storage((nameof(f), :bindgroup, T, size(x))), UInt32[], 0, 99999)
	WGPUCore.dispatchWorkGroups(computePass, size(x)...)
	WGPUCore.endComputePass(computePass)
	WGPUCore.submit(gpuDevice.queue, [WGPUCore.finish(commandEncoder),])
end

function kernelFunc(funcExpr)
	if @capture(funcExpr, f_(x_))
		kernelfunc = quote
			function $f(x::WgpuArray{T, N}) where {T, N}
				# x = getproperty(Main, Symbol($x)) # TODO Main is limiting # TODO deal with array of inputs later
				y = similar(x) # TODO we can have multiple outputs (interesting too)
				$preparePipeline($f, x, y)
				$compute($f, x)
				return y
			end
		end
		return esc(kernelfunc)
	elseif 	@capture(funcExpr, function fname_(fargs__) where Targs__ fbody__ end)
		kernelfunc = quote
			function $fname(x::WgpuArray{T, N}) where {T, N}
				y = similar(x)
				$preparePipeline($(funcExpr), x, y)
				$compute($(funcExpr), x)
				return y
			end
		end
		return esc(kernelfunc)
	else
		error("Couldnt capture function")
	end
end

macro kernel(expr)
	kernelFunc(expr)
end
