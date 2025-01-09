using LinearAlgebra

function sigmoid(x::VecOrMat{T})::VecOrMat{T} where T <: Real
    return 1 ./ (1 .+ exp.(-x))
end

function relu(x::VecOrMat{T})::VecOrMat{T} where T <: Real
    return max.(0, x)
end

function softmax(x::VecOrMat{T})::VecOrMat{T} where T <: Real
    x = x .- maximum(x, dims = ndims(x))   # オーバーフロー対策
    return exp.(x) ./ sum(exp.(x), dims = ndims(x))
end


if abspath(PROGRAM_FILE) == @__FILE__
    matrix::Matrix{Float32} = [1.0f0 2.0f0 3.0f0; 4.0f0 5.0f0 6.0f0]
    println(sigmoid(matrix)) # [[0.7310586 0.880797 0.95257413] [0.98201376 0.9933072 0.9975274]]
    println(relu(matrix)) # [[1 2 3] [4 5 6]]
    println(softmax(matrix)) # [[0.09003057 0.24472847 0.66524096] [0.09003057 0.24472847 0.66524096]]
end