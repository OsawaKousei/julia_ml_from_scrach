using LinearAlgebra

function mse(t::Matrix{Float32}, y::Matrix{Float32})::Float32
    batch_size = size(t, 2)
    return sum((t .- y) .^ 2) / batch_size
end

function cross_entropy_error(y::Matrix{Float32}, t::Matrix{Float32})::Float32
    if ndims(y) == 1
        t = reshape(t, 1, length(t))
        y = reshape(y, 1, length(y))
    end

    if length(t) == length(y)
        t = vec(getindex.(argmax(t, dims = 2), 2))  # 修正箇所
    else
        t = vec(t)
    end

    batch_size = size(y, 1)
    selected_y = [y[i, t[i]] for i in 1:batch_size]
    return -sum(log.(selected_y .+ 1e-7)) / batch_size
end

if abspath(PROGRAM_FILE) == @__FILE__
    t = [0.0f0 0.0f0 1.0f0 0.0f0 0.0f0; 0.0f0 0.0f0 0.0f0 0.0f0 1.0f0]
    y = [0.1f0 0.0f0 0.6f0 0.0f0 0.0f0; 0.1f0 0.05f0 0.0f0 0.0f0 0.05f0]
    # println(mse(t, y))
    println(cross_entropy_error(y, t))
end