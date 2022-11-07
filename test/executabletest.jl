using GLFW
using GLFW: Window, WindowShouldClose, PollEvents, DestroyWindow
using StaticCompiler
using StaticTools

function main(argc::Int, argv::Ptr{Ptr{UInt8}})
	window = Window("Hello", 500, 50)
	while !(WindowShouldClose(window))
	
	end
	DestroyWindow(window)
end

compile_executable(main, (Int64, Ptr{Ptr{UInt8}}), "./")

using GLFW_jll
using GLFW
using GLFW: Window, WindowShouldClose, PollEvents, DestroyWindow
using StaticCompiler
using StaticTools

function testy(argc::Int, argv::Ptr{Ptr{UInt8}})
	window = Window(name="Hello", resolution=(500, 500))
	while !(WindowShouldClose(window))
	
	end
	DestroyWindow(window)
end

filepath = compile_executable(
	testy, 
	(Int64, Ptr{Ptr{UInt8}}), 
	"./";
	cflags=`-lglfw -L $(dirname(libglfw)) -ljulia -L"/Users/arhik/.julia/juliaup/julia-1.8.2+0.aarch64/lib"`
)

using StaticTools

function print_args(argc::Int, argv::Ptr{Ptr{UInt8}})
    printf(c"Argument count is %d:\n", argc)
    for i=1:argc
        pᵢ = unsafe_load(argv, i) # Get pointer
        strᵢ = MallocString(pᵢ) # Can wrap to get high-level interface
        println(strᵢ)
    end
    println(c"That was fun, see you next time!")
    return 0
end

using StaticCompiler
filepath = compile_executable(print_args, (Int64, Ptr{Ptr{UInt8}}), "./")


using StaticCompiler
using StaticTools

function testy(argc::Int, argv::Ptr{Ptr{UInt8}})
	total::Int64 = 0
	for i in 1:argc
		tmp = argparse(Int64, argv, i)
		total += tmp
	end
	printf(c"Total : %d", total)
	return 0
end

filepath = compile_executable(testy, (Int64, Ptr{Ptr{UInt8}}), "./")

