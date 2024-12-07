using LinearAlgebra
include("activation_function.jl")
include("loss_function.jl")

mutable struct Affine
    W::Matrix{Float64}
    b::Vector{Float64}
    x::Matrix{Float64}
    dW::Matrix{Float64}
    db::Vector{Float64}
end

mutable struct Relu
    mask::Matrix{Bool}
end

mutable struct SoftmaxWithLoss
    loss::Float64
    y::Matrix{Float64}
    t::Matrix{Float64}
end

function Affine(W::Matrix{Float64}, b::Vector{Float64})
    return Affine(
        W,
        b,
        zeros(Float64, 0, 0),
        zeros(Float64, size(W)),
        zeros(Float64, size(b)),
    )
end

function Relu()
    return Relu(zeros(Bool, 0, 0))
end

function SoftmaxWithLoss()
    return SoftmaxWithLoss(0.0, zeros(Float64, 0, 0), zeros(Float64, 0, 0))
end

function forward(affine::Affine, x::Matrix{Float64})
    affine.x = x

    return x * affine.W .+ affine.b'
end

function forward(
    softmax_with_loss::SoftmaxWithLoss,
    x::Matrix{Float64},
    t::Matrix{Float64},
)
    softmax_with_loss.t = t
    softmax_with_loss.y = softmax(x)
    softmax_with_loss.loss =
        loss_function(softmax_with_loss.y, softmax_with_loss.t)
    return softmax_with_loss.loss
end

function forward(relu::Relu, x::Matrix{Float64})
    relu.mask = x .<= 0
    out = deepcopy(x)
    out[relu.mask] .= 0
    return out
end

function backward(affine::Affine, d_out::Matrix{Float64})
    dx = d_out * transpose(affine.W)
    affine.dW = transpose(affine.x) * d_out
    affine.db = vec(sum(d_out, dims = 1))

    return dx
end

function backward(relu::Relu, d_out::Matrix{Float64})
    d_out[relu.mask] .= 0
    return d_out
end

function backward(softmax_with_loss::SoftmaxWithLoss, d_out::Int)
    batch_size = size(softmax_with_loss.t, 1)
    return (softmax_with_loss.y .- softmax_with_loss.t) / batch_size
end