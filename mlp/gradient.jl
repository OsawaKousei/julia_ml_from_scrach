using LinearAlgebra
using ForwardDiff
using .Threads

function numerical_gradient_auto_diff(
    f::Function,
    x::VecOrMat{Float64},
)::VecOrMat{Float64}
    return ForwardDiff.gradient(f, x)
end

function numerical_gradient(f::Function, x::Array{Float64})::Array{Float64}
    h = 1e-4
    grad = zeros(size(x))


    for idx in CartesianIndices(x)
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2h)
        x[idx] = tmp_val  # 値を元に戻す
    end

    return grad
end

if abspath(PROGRAM_FILE) == @__FILE__
    f(x) = sum(x .^ 2)
    println(numerical_gradient_auto_diff(f, [3.0, 4.0])) # [6.0, 8.0]
    f(x) = sum(x .^ 2)
    println(numerical_gradient_auto_diff(f, [3.0 4.0; 5.0 6.0])) # [6.0 8.0; 10.0 12.0]
end