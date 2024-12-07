using LinearAlgebra

function mse(y::Matrix{Float64}, t::Matrix{Float64})::Float64
    batch_size = size(t, 2)
    return sum((t .- y) .^ 2) / batch_size
end

function loss_function(y::Matrix{Float64}, t::Matrix{Float64})::Float64
    batch_size = size(t, 1)
    # tを正解ラベルのindexに変換
    t = [argmax(t[i, :]) for i in 1:batch_size]

    # 予測値yから正解ラベルの値を取得
    y = [y[i, t[i]] for i in 1:batch_size]

    t = Float64.(t)

    return -sum(log.(y)) / batch_size
end

function cross_entropy_error(y::Vector{Float64}, t::Vector{Float64})::Float64
    delta = 1e-7
    return -sum(t .* log.(y .+ delta))
end

function cross_entropy_error(y::Matrix{Float64}, t::Matrix{Float64})::Float64
    delta = 1e-7
    batch_size = size(y, 1)
    return -sum(t .* log.(y .+ delta)) / batch_size
end

if abspath(PROGRAM_FILE) == @__FILE__
    t = [0.0 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0]
    y = [0.1 0.0 0.6 0.0 0.0; 0.1 0.05 0.0 0.0 0.05]
    # println(mse(t, y))
    println(loss_function(y, t)) # 1.7532778653276644
end