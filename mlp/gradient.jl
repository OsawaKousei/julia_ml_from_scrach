using LinearAlgebra

function numerical_gradient(f::Function, x::Matrix{Float32})::Matrix{Float32}
    h = 1e-4
    grad = zeros(size(x))
    for i in 1:size(x, 1)
        for j in 1:size(x, 2)
            tmp_val = x[i, j]
            x[i, j] = tmp_val + h
            fxh1 = f(x)
            x[i, j] = tmp_val - h
            fxh2 = f(x)
            grad[i, j] = (fxh1 - fxh2) / (2 * h)
            x[i, j] = tmp_val
        end
    end
    return grad
end

function numerical_gradient(f::Function, x::Vector{Float32})::Vector{Float32}
    h = 1e-4
    grad = zeros(size(x))
    for i in 1:length(x)
        tmp_val = x[i]
        x[i] = tmp_val + h
        fxh1 = f(x)
        x[i] = tmp_val - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp_val
    end
    return grad
end

if abspath(PROGRAM_FILE) == @__FILE__
    f(x::Vector{Float32}) = sum(x .^ 2)
    println(numerical_gradient(f, [3.0, 4.0]))
    f(x::Vector{Int64}) = sum(x .^ 2)
    println(numerical_gradient(f, [3, 4]))
end