export @wgpu, wgpu, emitWGSLJuliaBody

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
using Revise
using CodeTracking
using Lazy

# TODO remove
using WGPUCompute

# x = WgpuArray(rand(10, 10, 3) .|> Float32);

function wgpu(f, x)
	T = eltype(x)
	fexpr = @code_expr(f(x))
	
	# @capture(expr, function fdecl__ end) || error("Couldnt capture function")
	# @capture(fdecl[1], fname_(fargs__) where Targs_) || error("Couldnt function signature")
	
	@capture(fexpr, function fname_(fargs__) where Targs_ fbody__ end)
	
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

macro wgpu(expr, x)
	wgpu(expr, x)
end

mutable struct KernelContext
	inargs::Dict{Symbol, Any}
	outargs::Dict{Symbol, Any}
	tmpargs::Array{Symbol}
	typeargs::Array{Symbol}
	stmnts::Array{Expr}
	globals::Array{Expr}
	indent::Int
end

# @forward KernelContext.args push!

function emitWGSLJuliaBody(fbody, inargs)
	ins = Dict{Symbol, Any}()
	outs = Dict{Symbol, Any}()

	cntxt = KernelContext(ins, outs, Symbol[], Symbol[], Expr[], Expr[], 0)
	
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
	if @capture(fbody[end], a_=b_)
		idx = length(ins)
		iovar = Symbol(:ouput, idx)
		outs[a] = iovar
		push!(cntxt.globals, quote
			@var StorageReadWrite 0 $(idx) $(iovar)::IOArray
		end)
	elseif false
		# TODO others like return
		# TODO others like just symbol
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

function resolveInOutVars(cntxt, expr)
	
end

function wgslFunctionStatement(cntxt::KernelContext, stmnt; isLast = false)
	if @capture(stmnt, a_ = b_)
		if (a in cntxt.tmpargs) && !(a in cntxt.inargs |> keys) && !(a in cntxt.outargs |> keys)
			stmnt = :($(wgslFunctionStatement(a)) = $(wgslFunctionStatement(b)))
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
			return cntxt.inargs[stmnt]
		elseif (stmnt in cntxt.outargs |> keys)
			iovar = Symbol(:output, length(cntxt.outargs |> keys) + 1)
			return cntxt.outargs[stmnt]
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

