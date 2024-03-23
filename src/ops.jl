for (root, dirs, files) in walkdir(joinpath(@__DIR__, "ops"))
	for file in files
		include(joinpath(root, file))
	end
end
