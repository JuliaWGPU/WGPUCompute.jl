export @wgpu, wgpu

"""
    @wgpu [kwargs...] func(args...)

High-level interface for executing code on a GPU. The `@metal` macro should prefix a call,
with `func` a callable function or object that should return nothing. It will be compiled to
a Metal function upon first use, and to a certain extent arguments will be converted and
managed automatically using `wgpuconvert`. Finally, a call to `wgpucall` is
performed, creating a command buffer in the current global command queue then committing it.

There is one supported keyword argument that influences the behavior of `@metal`.
- `launch`: whether to launch this kernel, defaults to `true`. If `false` the returned
  kernel object should be launched by calling it and passing arguments again.
"""

using MacroTools
using Revise
using CodeTracking

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

# src = wgpu(Relu, x)


# using WGSLTypes
# using MacroTools
# 
# using GeometryBasics: Vec2, Vec3, Vec4, Mat4, Mat3, Mat2
# 
# export @code_wgsl
# 
# macro user(expr)
	# getproperty(@__MODULE__, expr)
# end
# 

# # TODO this function takes block of fields too
# # Another function that makes a sequence of field
# # statements is needed.
# function evalStructField(fieldDict, field)
	# if @capture(field, if cond_ ifblock__ end)
		# if eval(cond) == true
			# for iffield in ifblock
				# evalStructField(fieldDict, iffield)
			# end
		# end
	# elseif @capture(field, if cond_ ifblock__ else elseBlock__ end)
		# if eval(cond) == true
			# for iffield in ifblock
				# evalStructField(fieldDict, iffield)
			# end
		# else
			# for elsefield in elseBlock
				# evalStructField(fieldDict, elsefield)
			# end
		# end
	# elseif @capture(field, name_::dtype_)
		# return merge!(fieldDict, Dict(name=>eval(dtype)))
	# elseif @capture(field, @builtin btype_ name_::dtype_)
		# return merge!(fieldDict, Dict(name=>eval(:(@builtin $btype $dtype))))
	# elseif @capture(field, @location btype_ name_::dtype_)
		# return merge!(fieldDict, Dict(name=>eval(:(@location $btype $dtype))))
	# elseif @capture(field, quote stmnts__ end)
		# for stmnt in stmnts
			# evalStructField(fieldDict, stmnt)
		# end
	# else
		# @error "Unknown struct field! $field"
	# end
# end
# 
# function wgslStruct(expr)
	# expr = MacroTools.striplines(expr)
	# expr = MacroTools.flatten(expr)
	# @capture(expr, struct T_ fields__ end) || error("verify struct format of $T with fields $fields")
	# fieldDict = Dict{Symbol, DataType}()
	# for field in fields
		# evalfield = evalStructField(fieldDict, field)
	# end
	# makePaddedStruct(T, :UserStruct, sort(fieldDict))
	# makePaddedWGSLStruct(T, sort(fieldDict))
# end
# 
# # TODO rename simple asssignment and bring back original assignment if needed
# function wgslAssignment(expr)
	# io = IOBuffer()
	# @capture(expr, a_ = b_) || error("Expecting simple assignment a = b")
	# write(io, "$(wgslType(a)) = $(wgslType(b));\n")
	# seek(io, 0)
	# stmnt = read(io, String)
	# close(io)
	# return stmnt
# end
# 
# 
# function wgslFunctionStatement(io, stmnt)
	# if @capture(stmnt, @var t__)
		# write(io, " "^4*wgslVariable(stmnt))
	# elseif @capture(stmnt, a_ = b_)
		# write(io, " "^4*wgslAssignment(stmnt))
	# elseif @capture(stmnt, @let t_ | @let t__)
		# stmnt.args[1] = Symbol("@letvar") # replace let with letvar
		# write(io, " "^4*wgslLet(stmnt))
	# elseif @capture(stmnt, return t_)
		# write(io, " "^4*"return $(wgslType(t));\n")
	# elseif @capture(stmnt, if cond_ ifblock__ end)
		# if cond == true
			# wgslFunctionStatements(io, ifblock)
		# end
	# elseif @capture(stmnt, if cond_ ifBlock__ else elseBlock__ end)
		# if eval(cond) == true
			# wgslFunctionStatements(io, ifBlock)
		# else
			# wgslFunctionStatements(io, elseBlock)
		# end
	# else
		# @error "Failed to capture statment : $stmnt !!"
	# end
# end
# 
# function wgslFunctionStatements(io, stmnts)
	# for stmnt in stmnts
		# wgslFunctionStatement(io, stmnt)
	# end
# end
# 
# function wgslFunctionBody(fnbody, io, endstring)
	# if @capture(fnbody[1], fnname_(fnargs__)::fnout_)
		# write(io, "fn $fnname(")
		# len = length(fnargs)
		# endstring = len > 0 ? "}\n" : ""
		# for (idx, arg) in enumerate(fnargs)
			# if @capture(arg, aarg_::aatype_)
				# intype = wgslType(eval(aatype))
				# write(io, "$aarg:$(intype)"*(len==idx ? "" : ", "))
			# elseif @capture(arg, @builtin e_ => id_::typ_)
				# intype = wgslType(eval(typ))
				# write(io, "@builtin($e) $id:$(intype)")
			# end
			# @capture(fnargs, aarg_) || error("Expecting type for function argument in WGSL!")
		# end
		# outtype = wgslType(eval(fnout))
		# write(io, ") -> $outtype { \n")
		# @capture(fnbody[2], stmnts__) || error("Expecting quote statements")
		# wgslFunctionStatements(io, stmnts)
	# elseif @capture(fnbody[1], fnname_(fnargs__))
		# write(io, "fn $fnname(")
		# len = length(fnargs)
		# endstring = len > 0 ? "}\n" : ""
		# for (idx, arg) in enumerate(fnargs)
			# if @capture(arg, aarg_::aatype_)
				# intype = wgslType(eval(aatype))
				# write(io, "$aarg:$(intype)"*(len==idx ? "" : ", "))
			# elseif @capture(arg, @builtin e_ => id_::typ_)
				# intype = wgslType(eval(typ))
				# write(io, "@builtin($e) $id:$(intype)")
			# end
			# @capture(fnargs, aarg_) || error("Expecting type for function argument in WGSL!")
		# end
		# write(io, ") { \n")
		# @capture(fnbody[2], stmnts__) || error("Expecting quote statements")
		# wgslFunctionStatements(io, stmnts)
	# end
	# write(io, endstring)
# end
# 
# 
# 
# function wgslVertex(expr)
	# io = IOBuffer()
	# endstring = ""
	# @capture(expr, @vertex function fnbody__ end) || error("Expecting regular function!")
	# write(io, "@stage(vertex) ") # TODO should depend on version
	# wgslFunctionBody(fnbody, io, endstring)
	# seek(io, 0)
	# code = read(io, String)
	# close(io)
	# return code
# end
# 
# function wgslFragment(expr)
	# io = IOBuffer()
	# endstring = ""
	# @capture(expr, @fragment function fnbody__ end) || error("Expecting regular function!")
	# write(io, "@stage(fragment) ") # TODO should depend on version
	# wgslFunctionBody(fnbody, io, endstring)
	# seek(io, 0)
	# code = read(io, String)
	# close(io)
	# return code
# end
# 
# function wgslCompute(expr)
	# io = IOBuffer()
	# endstring = ""
	# if @capture(expr, @compute @workgroupSize(x_) function fnbody__ end)
		# write(io, "@stage(compute) @workgroup_size($x) \n")
	# elseif	@capture(expr, @compute @workgroupSize(x_,) function fnbody__ end)
		# write(io, "@stage(compute) @workgroup_size($x) \n")
	# elseif @capture(expr, @compute @workgroupSize(x_, y_) function fnbody__ end)
		# write(io, "@stage(compute) @workgroup_size($x, $y) \n")
	# elseif @capture(expr, @compute @workgroupSize(x_, y_, z_) function fnbody__ end)
		# write(io, "@stage(compute) @workgroup_size($x, $y, $z) \n")
	# else
		# error("Did not match the compute declaration function!")
	# end
	# wgslFunctionBody(fnbody, io, endstring)
	# seek(io, 0)
	# code = read(io, String)
	# close(io)
	# return code
# end
# 
# function wgslFunction(expr)
	# io = IOBuffer()
	# endstring = ""
	# @capture(expr, function fnbody__ end) || error("Expecting regular function!")
	# wgslFunctionBody(fnbody, io, endstring)
	# seek(io, 0)
	# code = read(io, String)
	# close(io)
	# return code
# end
# 
# function wgslVariable(expr)
	# io = IOBuffer()
	# write(io, wgslType(eval(expr)))
	# seek(io, 0)
	# code = read(io, String)
	# close(io)
	# return code
# end
# 
# # TODO for now both wgslVariable and wgslLet are same
# function wgslLet(expr)
	# io = IOBuffer()
	# write(io, wgslType(eval(expr)))
	# seek(io, 0)
	# code = read(io, String)
	# close(io)
	# return code
# end
# 
# # IOContext TODO
# function wgslCode(expr)
	# io = IOBuffer()
	# expr = MacroTools.striplines(expr)
	# expr = MacroTools.flatten(expr)
	# @capture(expr, blocks__) || error("Current expression is not a quote or block")
	# for block in blocks
		# if @capture(block, struct T_ fields__ end)
			# write(io, wgslStruct(block))
		# elseif @capture(block, a_ = b_)
			# write(io, wgslAssignment(block))
		# elseif @capture(block, @var t__)
			# write(io, wgslVariable(block))
		# elseif @capture(block, @vertex function a__ end)
			# write(io, wgslVertex(block))
			# write(io, "\n")
		# elseif @capture(block, @compute @workgroupSize(x__) function a__ end)
			# write(io, wgslCompute(block))
			# write(io, "\n")
		# elseif @capture(block, @fragment function a__ end)
			# write(io, wgslFragment(block))
			# write(io, "\n")
		# elseif @capture(block, function a__ end)
			# write(io, wgslFunction(block))
			# write(io, "\n")
		# elseif @capture(block, if cond_ ifblock_ end)
			# if eval(cond) == true
				# write(io, wgslCode(ifblock))
				# write(io, "\n")
			# end
		# elseif @capture(block, if cond_ ifBlock_ else elseBlock_ end)
			# if eval(cond) == true
				# write(io, wgslCode(ifBlock))
				# write(io, "\n")
			# else
				# write(io, wgslCode(elseBlock))
				# write(io, "\n")
			# end
		# end
	# end
	# seek(io, 0)
	# code = read(io, String)
	# close(io)
	# return code
# end
# 
# macro code_wgsl(expr)
	# a = wgslCode(eval(expr)) |> println
	# return a
# end

