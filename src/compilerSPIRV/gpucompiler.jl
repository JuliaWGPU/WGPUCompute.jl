const ci_cache = GPUCompiler.CodeCache()

struct SPIRVCompilerParams <: GPUCompiler.AbstractCompilerParams end

SPIRVCompilerJob = CompilerJob{SPIRVCompilerTarget, SPIRVCompilerParams}

GPUCompiler.runtime_module(::SPIRVCompilerJob) = SPIRV

GPUCompiler.ci_cache(::SPIRVCompilerJob) = ci_cache

GPUCompiler.method_table(::SPIRVCompilerJob) = method_table
