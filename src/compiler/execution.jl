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
	expr = @code_expr(f(x))
	@capture(expr, function fdecl__ end) || error("Couldnt capture function")
	@capture(fdecl[1], fname_(fargs__) where Targs_) || error("Couldnt function signature")  

	# @capture(fexpr, function fname_(fargs__) where Targs_ fbody__ end)
	
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

	code = quote end

	# stmnts = emitWGSLJuliaBody(fbody, fargs)
	
	fquote = quote
		function $(fname)($(builtinArgs...))
			$((fdecl[2].args)...)
		end
	end
	
    push!(code.args,
		:(@compute @workgroupSize(8, 8, 4) $(fquote.args...))
   	)

    return code
end

macro wgpu(expr, x)
	wgpu(expr, x)
end

mutable struct WgpuContext
	inargs::Array{Symbol}
	outargs::Array{Symbol}
	tmpargs::Array{Symbol}
	stmnts::Array{Expr}
end

# @forward WgpuContext.args push!

function emitWGSLJuliaBody(fbody, inargs)
	ins = Symbol[]
	for arg in inargs
		@capture(arg, a_::b_)
		push!(ins, a)
	end
	cntxt = WgpuContext(ins, Symbol[], Symbol[], Expr[])
	wgslFunctionStatements(cntxt, fbody)
	cntxt
	
	# TODO infer outargs too
	# so that we can generate the necessary output structs with
	# appropriate readwrite operations
	
end

function wgslAssignment(expr::Expr, prefix::Union{Nothing, Symbol})
	@capture(expr, a_ = b_) || error("Expecting simple assignment a = b")
	return ifelse(prefix==:let, :(@let $a = $b), :($a = $b))
end

function wgslFunctionStatements(cntxt, stmnts)
	for stmnt in stmnts
		wgslFunctionStatement(cntxt, stmnt)
	end
end

function wgslFunctionStatement(cntxt::WgpuContext, stmnt)
	if @capture(stmnt, a_ = b_)
		if a in cntxt.tmpargs
			push!(cntxt.stmnts, wgslAssignment(stmnt, nothing))
		else
			push!(cntxt.stmnts, wgslAssignment(stmnt, :let))
			push!(cntxt.tmpargs, a)
		end
	elseif @capture(stmnt, @let t_ | @let t__)
		stmnt.args[1] = Symbol("@letvar") # replace let with letvar
		push!(cntxt.stmnts, wgslLet(stmnt))
	elseif @capture(stmnt, a_ += b_)
		wgslFunctionStatement(cntxt, :($a = $a + $b))
	elseif @capture(stmnt, a_ -= b_)
		wgslFunctionStatement(cntxt, :($a = $a - $b))
	elseif @capture(stmnt, a_ *= b_)
		wgslFunctionStatement(cntxt, :($a = $a * $b))
	elseif @capture(stmnt, a_ /= b_)
		wgslFunctionStatement(cntxt, :($a = $a / $b))
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

