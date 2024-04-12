# WGPUCompute

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaWGPU.github.io/WGPUCompute.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaWGPU.github.io/WGPUCompute.jl/dev/)
[![Build Status](https://github.com/JuliaWGPU/WGPUCompute.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaWGPU/WGPUCompute.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaWGPU/WGPUCompute.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaWGPU/WGPUCompute.jl)

:warning: This repo is under heavy development.

`WGPUCompute` is a `WGPU` compute shader utility library for julia. Using this library one can define compute shader kernels in regular julia. For example:

```julia 

using BenchmarkTools 
using WGPUCompute

# Kernel definition
function cast_kernel(x::WgpuArray{T, N}, out::WgpuArray{S, N}) where {T, S, N}
	xdim = workgroupDims.x
	ydim = workgroupDims.y
	gIdx = workgroupId.x*xdim + localId.x
	gIdy = workgroupId.y*ydim + localId.y
	gId = xDims.x*gIdy + gIdx
	out[gId] = S(ceil(x[gId]))
end

# wrapper function
function cast(S::DataType, x::WgpuArray{T, N}) where {T, N}
	y = WgpuArray{S}(undef, size(x))
	@wgpukernel launch=true workgroupSizes=(4, 4) workgroupCount=(2, 2) shmem=() cast_kernel(x, y)
	return y
end

x = WgpuArray{Float32}(rand(Float32, 8, 8) .- 0.5f0)
z = cast(UInt32, x)

```

In the above example single generalized kernel can be used for casting different datatypes. The type parameters `S`, `T`, & `N` are inferred and replaced with their actual type information internally.

Compute kernels also support defining shared memory and can provide means to implement kernels like matmul. For example


```julia
function tiled_matmul_kernel(x::WgpuArray{T, N}, y::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	#set out matrix to zero
	gId = xDims.x*globalId.y + globalId.x
	out[gId] = 0.0
	
	# set local variable = 0.0
	sum = 0.0
	
	for tileId in 0:numWorkgroups.y
		# copy block from x to shared memory
		xId = workgroupId.x*workgroupDims.x + localId.x
		yId = tileId*workgroupDims.y + localId.y
		sId = localId.y*workgroupDims.x + localId.x
		shmem1[sId] = x[yId*xDims.x + xId]
		
		# copy block from y to shared memory
		xId = tileId*workgroupDims.x + localId.x
		yId = workgroupId.y*workgroupDims.y + localId.y
		shmem2[sId] = y[yId*yDims.x + xId]
		synchronize()
				
		# block sums for each tid
		for i in 0:xDims.y/numWorkgroups.y
			sum = sum + shmem1[i*workgroupDims.x + localId.x]*shmem2[localId.y*workgroupDims.x + i]
		end
		synchronize()
	end
	
	out[gId] = sum
end

# For now valid only for square matrices of size powers of 2 and base size 16 to keep it simple.
function tiled_matmul_heuristics(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	aSize = size(x)
	bSize = size(y)
	@assert last(aSize) == first(bSize)
	outSize = (first(aSize), last(bSize))
	@assert eltype(x) == eltype(y)
	wgSize = (16, 16) # This can be fixed for now
	wgCount = div.((outSize[1], outSize[2]), 16, RoundUp)
	return (outSize, wgSize, wgCount)
end

function tiled_matmul(x::WgpuArray{T, N}, y::WgpuArray{T, N}) where {T, N}
	(outSize, wgSize, wgCount) = tiled_matmul_heuristics(x, y)
	out = WgpuArray{eltype(x), ndims(x)}(undef, outSize)
	@wgpukernel(
		launch=true,
		workgroupSizes=wgSize,
		workgroupCount=wgCount,
		shmem=(:shmem1=>(Float32, wgSize), :shmem2=>(Float32, wgSize)),
		tiled_matmul_kernel(x, y, out)
	)
	return out
end

Base.:*(x::WgpuArray{T, N}, y::WgpuArray{T, N})  where {T, N} = tiled_matmul(x, y)

x = WgpuArray{Float32, 2}(rand(2048, 2048));
y = WgpuArray{Float32, 2}(rand(2048, 2048));

z = x*y

z_cpu = (x |> collect)*(y |> collect)

@test z_cpu ≈ (z |> collect)



```

There is limited supported for GPUArrays interface. And is currently under development to make is complete.

```julia
using WGPUCompute
using BenchmarkTools

aArray = WgpuArray{Float32}(undef, (1024, 1024, 100)) 
bArray = WgpuArray{Float32}(rand(Float32, (1024, 1024, 100)))

@benchmark copyto!(aArray, 1, bArray, 1, prod(size(aArray)))

```
```
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  62.900 μs …  1.885 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     70.100 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   95.964 μs ± 80.628 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

   ▇█▄▃▁▁▃▃▂▂▂▂▂▂▁▂▂▁▁  ▁▂▃▂  ▁▁▂▃▃▂  ▁▂▁▂▁                   ▂
  █████████████████████████████████████████▇▆▆▅▅▅▇█▇▆▆▇▇▇▆▅▆▆ █
  62.9 μs      Histogram: log(frequency) by time       208 μs <

 Memory estimate: 1.01 KiB, allocs estimate: 37.
 ```

Basic ML kernels can be defined:

A very simplified kernel example of ML primitive `relu`:

```julia
using WGPUCompute

y = WgpuArray((rand(4, 4) .-0.5) .|> Float32)

function relu_kernel(x::WgpuArray{T, N}, out::WgpuArray{T, N}) where {T, N}
	gId = xDims.x*globalId.y + globalId.x
	value = x[gId]
	out[gId] = max(value, 0.0)
end

function relu(x::WgpuArray{T, N}) where {T, N}
	y = similar(x)
	@wgpukernel launch=true workgroupSizes=(4,4) workgroupCount=(1,1) shmem=() relu_kernel(x, y)
	return y
end

relu(y)

```

The above kernel undergoes two transformations:
1. First the `@wgpukernel` kernel macro takes the kernel function and transforms into an custom AST and intermeditate representation. This transformation is actually carried out the work done in `WGPUTranspiler`. And this AST is again transpiled to the below format. This is very close to `WGSL` but with julia IR semantics. For more detailed explanation please browse to this [link](https://github.com/JuliaWGPU/WGPUTranspier.jl).
```
┌ Info: begin
│     @const workgroupDims = Vec3{UInt32}(0x00000004, 0x00000004, 0x00000001)
│     @const xDims = Vec3{UInt32}(0x00000004, 0x00000004, 0x00000001)
│     @const outDims = Vec3{UInt32}(0x00000004, 0x00000004, 0x00000001)
│     @var StorageReadWrite 0 0 x::Array{Float32, 16}
│     @var StorageReadWrite 0 1 out::Array{Float32, 16}
│     @compute @workgroupSize(4, 4, 1) function relu_kernel(@builtin(global_invocation_id, globalId::Vec3{UInt32}), @builtin(local_invocation_id, localId::Vec3{UInt32}), @builtin(num_workgroups, numWorkgroups::Vec3{UInt32}), @builtin(workgroup_id, workgroupId::Vec3{UInt32}))
│             @let gId = xDims.x * globalId.y + globalId.x
│             @let value = x[gId]
│             out[gId] = max(value, 0.0f0)
│         end
└ end
```
2. Then this representation is again compiled to webgpu/WGPU's representation, `WGSL`. This is carried out an another package called `WGSLTypes`. 

```
┌ Info: const workgroupDims = vec3<u32>(4u, 4u, 1u);
│ const xDims = vec3<u32>(4u, 4u, 1u);
│ const outDims = vec3<u32>(4u, 4u, 1u);
│ @group(0) @binding(0) var<storage, read_write> x:array<f32, 16> ;
│ @group(0) @binding(1) var<storage, read_write> out:array<f32, 16> ;
│ @compute @workgroup_size(4, 4, 1) 
│ fn relu_kernel(@builtin(global_invocation_id) globalId:vec3<u32>, @builtin(local_invocation_id) localId:vec3<u32>, @builtin(num_workgroups) numWorkgroups:vec3<u32>, @builtin(workgroup_id) workgroupId:vec3<u32>) { 
│     let gId = xDims.x * globalId.y + globalId.x;
│     let value = x[gId];
│     out[gId] = max(value, 0.0);
│ }
└ 
```

This final shader code is compiled using `naga`, `WGPU-native`'s compiler.

## Conventions

1. Input arguments are converted into `storage` variables and placed at the top of the shader code.
2. Size of input arguments are converted into `const` variables and placed at the top of the shader code. Users can use these arguments to probe for input arrays's size. The corresponding name of variable declaring size of array will be  a concatenation of variable name followed by "Dims". For example: if variable is `x`, `xDims` holds the size information.  
3. Kernel arguments like `workgroupDims` etc are also placed at the top of the shader code and can be used as an variables inside kernel code. This will eventually be probed using julia's `size` function. Until then we can use this convention.
4. Shared memory can be declared in the `@wgpukernel` macro using `shmem` kwarg. `shmem` expects a tuple of pairs with each pair representing name and (type, size) of shared memory. Example: `shmem = ("xShared"=>(Float32, 16))`



## Known issues

- jupyter notebooks are not tested yet and might need some work to have compatibility with pluto as well.

## TODO

- [ ] atomics support is under development.
- [ ] possibility of JSServe the generated wgsl code in web app.
- [ ] Complete SPIRV version
- [ ] Explore and adhere to Binary generation eventually. 
