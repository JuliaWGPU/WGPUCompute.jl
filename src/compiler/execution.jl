export @wgpu, wgpu

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

	originArgs = fargs[:]

	builtinArgs = [:(@builtin global_invocation_id => global_id::Vec3{UInt32})]

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

	fquote = quote
		function $(fname)($(builtinArgs...)) where $(Targs)
			$((fdecl[2].args)...)
		end
	end
	
    push!(code.args,
		:(@compute @workgroupSize(8, 8, 4) $(fquote.args...))
   	)

    return code
end


mutable struct WgpuContext
	args::Array{Symbol}
end

@forward WgpuContext.args push! 

function emitWGSLJuliaBody(fbody)
	cntxt = WgpuContext(Symbol[])
	wgslFunctionStatements(cntxt, fbody)
end


function wgslAssignment(expr::Expr, prefix::Union{Nothing, Symbol})
	@capture(expr, a_ = b_) || error("Expecting simple assignment a = b")
	write(io, "$(wgslType(a)) = $(wgslType(b));\n")
	seek(io, 0)
	stmnt = read(io, String)
	close(io)
	return stmnt
end


function wgslFunctionStatements(cntxt, stmnts)
	for stmnt in stmnts
		wgslFunctionStatement(cntxt, stmnt)
	end
end


function wgslFunctionStatement(cntxt::WgpuContext, stmnt)
	if @capture(stmnt, a_ = b_)
		if a in cntxt.args
			# if a is not seen before then it must be local
			# TODO check if its referring to global variable
			# or input/output variable
			wglsAssignment(stmnt, :let)
		else
			push!(cntxt, " "^4*wgslAssignment(stmnt, nothing))
		end
	elseif @capture(stmnt, @let t_ | @let t__)
		stmnt.args[1] = Symbol("@letvar") # replace let with letvar
		push!(cntxt, " "^4*wgslLet(stmnt))
	elseif @capture(stmnt, return t_)
		push!(cntxt, " "^4*"return $(wgslType(t));\n")
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

