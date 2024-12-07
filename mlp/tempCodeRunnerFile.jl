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