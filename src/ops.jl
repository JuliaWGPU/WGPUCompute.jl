include("ops/matmul.jl")
include("ops/transpose.jl")
include("ops/cast.jl")
include("ops/clamp.jl")
include("ops/elemwise.jl")

"""
for (root, dirs, files) in walkdir(joinpath(@__DIR__, "ops"))
	for file in files
		include(joinpath(root, file))
	end
end
"""
