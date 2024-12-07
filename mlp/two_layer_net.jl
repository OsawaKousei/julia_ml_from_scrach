using LinearAlgebra
using MLDatasets
using DataStructures

include("activation_function.jl")
include("loss_function.jl")
include("layer.jl")
include("gradient.jl")

mutable struct TwoLayerNet
    params::Dict{String, Union{Matrix{Float64}, Vector{Float64}}}
    layers::OrderedDict{String, Union{Affine, Relu, SoftmaxWithLoss}}
    last_layer::SoftmaxWithLoss
end

function TwoLayerNet(
    input_size::Int,
    hidden_size::Int,
    output_size::Int,
    std::Float64 = 0.01,
)::TwoLayerNet
    net = TwoLayerNet(
        Dict{String, Union{Matrix{Float64}, Vector{Float64}}}(),
        OrderedDict{String, Union{Affine, Relu, SoftmaxWithLoss}}(),
        SoftmaxWithLoss(),
    )
    net.params = Dict{String, Union{Matrix{Float64}, Vector{Float64}}}()
    net.params["W1"] = std * randn(Float64, input_size, hidden_size)
    net.params["b1"] = zeros(Float64, hidden_size)
    net.params["W2"] = std * randn(Float64, hidden_size, output_size)
    net.params["b2"] = zeros(Float64, output_size)

    net.layers = OrderedDict{String, Union{Affine, Relu, SoftmaxWithLoss}}()
    net.layers["Affine1"] = Affine(net.params["W1"], net.params["b1"])
    net.layers["Relu1"] = Relu()
    net.layers["Affine2"] = Affine(net.params["W2"], net.params["b2"])

    net.last_layer = SoftmaxWithLoss()

    return net
end

function TwoLayerNet(
    W1::Matrix{Float64},
    W2::Matrix{Float64},
    b1::Vector{Float64},
    b2::Vector{Float64},
)::TwoLayerNet
    net = TwoLayerNet(
        Dict{String, Union{Matrix{Float64}, Vector{Float64}}}(),
        OrderedDict{String, Union{Affine, Relu, SoftmaxWithLoss}}(),
        SoftmaxWithLoss(),
    )
    net.params = Dict{String, Union{Matrix{Float64}, Vector{Float64}}}()
    net.params["W1"] = W1
    net.params["b1"] = b1
    net.params["W2"] = W2
    net.params["b2"] = b2

    return net
end

function predict(net::TwoLayerNet, x::Matrix{Float64})::Matrix{Float64}
    for (_, layer) in net.layers
        x = forward(layer, x)
    end
    return x
end

function loss(net::TwoLayerNet, x::Matrix{Float64}, t::Matrix{Float64})::Float64
    y = predict(net, x)
    return forward(net.last_layer, y, t)
end

function accuracy(
    net::TwoLayerNet,
    x::Matrix{Float64},
    t::Matrix{Float64},
)::Float64
    y = predict(net, x)
    y = vec(argmax(y, dims = 2))

    if ndims(t) != 1
        t = vec(argmax(t, dims = 2))
    end

    accuracy = sum(y .== t) / Float64(length(y))
    return accuracy
end

function numerical_gradient(
    net::TwoLayerNet,
    x::Matrix{Float64},
    t::Matrix{Float64},
)::Dict{String, Union{Matrix{Float64}, Vector{Float64}}}

    grads = Dict{String, Union{Matrix{Float64}, Vector{Float64}}}()

    func_W1(W1)::Float64 = loss(
        TwoLayerNet(W1, net.params["W2"], net.params["b1"], net.params["b2"]),
        x,
        t,
    )
    grads["W1"] = numerical_gradient_auto_diff(func_W1, net.params["W1"])
    func_b1(b1)::Float64 = loss(
        TwoLayerNet(net.params["W1"], net.params["W2"], b1, net.params["b2"]),
        x,
        t,
    )
    grads["b1"] = numerical_gradient_auto_diff(func_b1, net.params["b1"])
    func_W2(W2)::Float64 = loss(
        TwoLayerNet(net.params["W1"], W2, net.params["b1"], net.params["b2"]),
        x,
        t,
    )
    grads["W2"] = numerical_gradient_auto_diff(func_W2, net.params["W2"])
    func_b2(b2)::Float64 = loss(
        TwoLayerNet(net.params["W1"], net.params["W2"], net.params["b1"], b2),
        x,
        t,
    )
    grads["b2"] = numerical_gradient_auto_diff(func_b2, net.params["b2"])

    return grads
end

function gradient(
    net::TwoLayerNet,
    x::Matrix{Float64},
    t::Matrix{Float64},
)::Dict{String, Union{Matrix{Float64}, Vector{Float64}}}
    # forward
    loss(net, x, t)

    # backward
    d_out = 1
    d_out = backward(net.last_layer, d_out)

    layers = collect(values(net.layers))
    layers = reverse(layers)
    for layer in layers
        d_out = backward(layer, d_out)
    end

    grads = Dict{String, Union{Matrix{Float64}, Vector{Float64}}}()
    grads["W1"] = net.layers["Affine1"].dW
    grads["b1"] = net.layers["Affine1"].db
    grads["W2"] = net.layers["Affine2"].dW
    grads["b2"] = net.layers["Affine2"].db

    return grads
end

function update_params(
    net::TwoLayerNet,
    grads::Dict{String, Union{Matrix{Float64}, Vector{Float64}}},
    learning_rate::Float64,
)
    for (key, _) in net.params
        net.params[key] -= learning_rate * grads[key]
    end

    net.layers["Affine1"].W = net.params["W1"]
    net.layers["Affine1"].b = net.params["b1"]
    net.layers["Affine2"].W = net.params["W2"]
    net.layers["Affine2"].b = net.params["b2"]

end

function to_one_hot(t::Vector{Int}, num_classes::Int)::Matrix{Float64}
    one_hot = zeros(Float64, length(t), num_classes)
    for i in 1:length(t)
        one_hot[i, t[i] + 1] = 1
    end

    return one_hot
end

if abspath(PROGRAM_FILE) == @__FILE__
    const iters::Int = 300
    const batch_size::Int = 100
    const learning_rate::Float64 = 0.10

    train_data = MLDatasets.MNIST(split = :train)
    test_data = MLDatasets.MNIST(split = :test)

    train_x, train_t = Float64.(train_data.features), train_data.targets
    test_x, test_t = Float64.(test_data.features), test_data.targets

    # 3次元配列を2次元配列に変換
    train_x =
        reshape(train_x, size(train_x, 3), size(train_x, 1) * size(train_x, 2))
    test_x = reshape(test_x, size(test_x, 3), size(test_x, 1) * size(test_x, 2))

    println("Size of train_x: ", size(train_x))
    println("Size of test_x: ", size(test_x))

    println("Size of train_t: ", size(train_t))
    println("Size of test_t: ", size(test_t))

    train_size = size(train_x, 1)
    println("Train size: ", train_size)
    loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = round(Int, train_size / batch_size)
    println("Iter per epoch: ", iter_per_epoch)

    net = TwoLayerNet(784, 50, 10)

    test_t = to_one_hot(test_t, 10)
    println("Size of test_x: ", size(test_x))
    println("Size of test_t: ", size(test_t))

    for i in 1:iters
        batch_mask = rand(1:train_size, batch_size)
        x_batch = train_x[batch_mask, :]
        t_batch = train_t[batch_mask]

        # t_batchをone-hot表現に変換
        t_batch = to_one_hot(t_batch, 10)

        # println("Size of x_batch: ", size(x_batch))
        # println("Size of t_batch: ", size(t_batch))

        grads = numerical_gradient(net, x_batch, t_batch)
        # grads = gradient(net, x_batch, t_batch)

        update_params(net, grads, learning_rate)

        loss_value = loss(net, x_batch, t_batch)

        push!(loss_list, loss_value)

        if i % iter_per_epoch * 0 == 0
            train_acc_value = accuracy(net, x_batch, t_batch)
            push!(train_acc_list, train_acc_value)

            test_acc_value = accuracy(net, test_x, test_t)
            push!(test_acc_list, test_acc_value)
            println(
                "iter",
                i,
                "loss",
                loss_value,
                " train acc, test acc | ",
                train_acc_value,
                ", ",
                test_acc_value,
            )
        end
    end
end
