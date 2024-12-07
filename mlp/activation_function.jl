using LinearAlgebra

function sigmoid(x::Matrix)
    return 1 ./ (1 .+ exp.(-x))
end

function relu(x::Matrix{Float32})::Matrix{Float32}
    return max.(0, x)
end

function softmax(x::Matrix{Float32})::Matrix{Float32}
    a = maximum(x, dims = 2)
    exp_x = exp.(x .- a)
    sum_exp_x = sum(exp_x)
    return exp_x ./ sum_exp_x
end


if abspath(PROGRAM_FILE) == @__FILE__
    matrix::Matrix{Float32} = [1.0f0 2.0f0 3.0f0; 4.0f0 5.0f0 6.0f0]
    println(matrix)
    println(relu(matrix))
    println(softmax(matrix))
end