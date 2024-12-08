using LinearAlgebra
using Random
using MLDatasets

mutable struct MLP
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
end