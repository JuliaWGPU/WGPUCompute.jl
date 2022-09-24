using WGSLTypes
using LinearAlgebra
using StaticArrays

abstract type ActivationLayer{T} <: AbstractLayer{T} end

Base.eltype(act::ActivationLayer{T}) where T =  T

function getShaderCode(activation::ActivationLayer{T}) where T
	@error "If you reached here. Raise bug"
end
