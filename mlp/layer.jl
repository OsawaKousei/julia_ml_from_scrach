using LinearAlgebra
include("activation_function.jl")
include("loss_function.jl")

mutable struct Affine
    W::Matrix{Float32}
    b::Vector{Float32}
    x::Matrix{Float32}
    dW::Matrix{Float32}
    db::Vector{Float32}
end

mutable struct Relu
    mask::Matrix{Bool}
end

mutable struct SoftmaxWithLoss
    loss::Float32
    y::Matrix{Float32}
    t::Matrix{Float32}
end

function Affine(W::Matrix{Float32}, b::Vector{Float32})
    return Affine(
        W,
        b,
        zeros(Float32, 0, 0),
        zeros(Float32, size(W)),
        zeros(Float32, size(b)),
    )
end

function Relu()
    return Relu(zeros(Bool, 0, 0))
end

function SoftmaxWithLoss()
    return SoftmaxWithLoss(0.0, zeros(Float32, 0, 0), zeros(Float32, 0, 0))
end

function forward(affine::Affine, x::Matrix{Float32})
    affine.x = x

    return x * affine.W .+ affine.b'
end

function forward(
    softmax_with_loss::SoftmaxWithLoss,
    x::Matrix{Float32},
    t::Matrix{Float32},
)
    softmax_with_loss.t = t
    softmax_with_loss.y = softmax(x)
    softmax_with_loss.loss =
        cross_entropy_error(softmax_with_loss.y, softmax_with_loss.t)
    return softmax_with_loss.loss
end

function forward(relu::Relu, x::Matrix{Float32})
    relu.mask = x .<= 0
    out = deepcopy(x)
    out[relu.mask] .= 0
    return out
end

function backward(affine::Affine, d_out::Matrix{Float32})
    dx = d_out * transpose(affine.W)
    affine.dW = transpose(affine.x) * d_out
    affine.db = vec(sum(d_out, dims = 1))

    return dx
end

function backward(relu::Relu, d_out::Matrix{Float32})
    d_out[relu.mask] .= 0
    return d_out
end

function backward(softmax_with_loss::SoftmaxWithLoss, d_out::Int)
    batch_size = size(softmax_with_loss.t, 1)
    return (softmax_with_loss.y .- softmax_with_loss.t) / batch_size
end