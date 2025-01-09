using LinearAlgebra

function mse(y::VecOrMat{T}, t::VecOrMat{T})::T where T <: Real
    batch_size = size(t, 1)
    return sum((t .- y) .^ 2) / batch_size
end

function loss_function(y::Matrix{T}, t::Matrix{T})::T where T <: Real
    batch_size = size(t, 1)
    # tを正解ラベルのindexに変換
    t = [argmax(t[i, :]) for i in 1:batch_size]

    # 予測値yから正解ラベルの値を取得
    y = [y[i, t[i]] for i in 1:batch_size]

    return -sum(log.(y.+ 1e-7)) / batch_size
end

function cross_entropy_error(y::VecOrMat{T}, t::VecOrMat{T})::T where T <: Real
    delta = 1e-7
    return -sum(t .* log.(y .+ delta))
end

if abspath(PROGRAM_FILE) == @__FILE__
    t = [0.0 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0]
    y = [0.1 0.0 0.6 0.0 0.0; 0.1 0.05 0.0 0.0 0.05]
    println(mse(t, y)) # 0.5425
    println(cross_entropy_error(y, t)) # 3.5065557306553288
    println(loss_function(y, t)) # 1.7532778653276644
end