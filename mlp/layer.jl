using LinearAlgebra
include("activation_function.jl")
include("loss_function.jl")

mutable struct Affine{T <: Real}
    W::Matrix{T}
    b::Vector{T}
    x::Matrix{T}
    dW::Matrix{T}
    db::Vector{T}
end

mutable struct Relu{T <: Real}
    mask::Matrix{Bool}
end

mutable struct SoftmaxWithLoss{T <: Real}
    loss::T
    y::Matrix{T}
    t::Matrix{T}
end

function Affine{T}(W::Matrix{T}, b::Vector{T}) where T <: Real
    return Affine(
        W,
        b,
        zeros(T, 0, 0),
        zeros(T, size(W)),
        zeros(T, size(b)),
    )
end

function Relu{T}() where T <: Real
    return Relu{T}(zeros(Bool, 0, 0))
end

function SoftmaxWithLoss{T}() where T <: Real
    return SoftmaxWithLoss(convert(T, 0.0), zeros(T, 0, 0), zeros(T, 0, 0))
end

function forward(affine::Affine, x::Matrix{T})::Matrix{T} where T <: Real
    affine.x = x

    return x * affine.W .+ affine.b'
end

function forward(
    softmax_with_loss::SoftmaxWithLoss,
    x::Matrix{T},
    t::Matrix{T},
)::T where T <: Real
    softmax_with_loss.t = t
    softmax_with_loss.y = softmax(x)
    softmax_with_loss.loss =
        loss_function(softmax_with_loss.y, softmax_with_loss.t)
    return softmax_with_loss.loss
end

function forward(relu::Relu, x::Matrix{T})::Matrix{T} where T <: Real
    relu.mask = x .<= 0
    out = deepcopy(x)
    out[relu.mask] .= convert(T, 0.0)
    return out
end

function backward(affine::Affine, d_out::Matrix{T})::Matrix{T} where T <: Real
    dx = d_out * transpose(affine.W)
    affine.dW = transpose(affine.x) * d_out
    affine.db = vec(sum(d_out, dims = 1))

    return dx
end

function backward(relu::Relu, d_out::Matrix{T}) where T <: Real
    d_out[relu.mask] .= convert(T, 0.0)
    return d_out
end

function backward(softmax_with_loss::SoftmaxWithLoss, d_out::T) where T <: Real
    batch_size = size(softmax_with_loss.t, 1)
    return (softmax_with_loss.y .- softmax_with_loss.t) / batch_size
end

if abspath(PROGRAM_FILE) == @__FILE__
    mat1 = [-1.0 0.0 1.0; 2.0 3.0 4.0]
    mat2 = [1.0 2.0; 3.0 4.0]

    relu_layer = Relu{Float32}()
    println(forward(relu_layer, mat1)) # [[0.0 0.0 1.0] [2.0 3.0 4.0]]
    println(backward(relu_layer, mat1)) # [[0.0 0.0 1.0] [2.0 3.0 4.0]]

    W = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    b = [1.0, 2.0]

    affine_layer = Affine{Float64}(W, b)
    println(forward(affine_layer, mat1)) # [[5.0 6.0] [32.0 42.0]
    println(backward(affine_layer, mat2)) # [[5.0 11.0 17.0] [11.0 25.0 39.0]]

    y = [0.0 0.75 0.5; 0.0 0.25 0.5]
    t = [0.0 1.0 0.0; 0.0 0.0 1.0]
    softmax_with_loss_layer = SoftmaxWithLoss{Float64}()
    println(forward(softmax_with_loss_layer, y, t)) # 0.8403932591751839
    println(backward(softmax_with_loss_layer, 1.0)) # [0.1049159130079824 -0.2778930104191673 0.17297709741118486; 0.12713760629523282 0.16324791789991835 -0.2903855241951512]
end