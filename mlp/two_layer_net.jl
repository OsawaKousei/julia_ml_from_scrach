using LinearAlgebra
using MLDatasets
using DataStructures

include("activation_function.jl")
include("loss_function.jl")
include("layer.jl")
include("gradient.jl")

mutable struct TwoLayerNet
    params::Dict{String, Union{Matrix{Float32}, Vector{Float32}}}
    layers::OrderedDict{String, Union{Affine, Relu, SoftmaxWithLoss}}
    last_layer::SoftmaxWithLoss
end

function TwoLayerNet(
    input_size::Int,
    hidden_size::Int,
    output_size::Int,
    std::Float32 = 0.01f0,
)::TwoLayerNet
    net = TwoLayerNet(
        Dict{String, Union{Matrix{Float32}, Vector{Float32}}}(),
        OrderedDict{String, Union{Affine, Relu, SoftmaxWithLoss}}(),
        SoftmaxWithLoss(),
    )
    net.params = Dict{String, Union{Matrix{Float32}, Vector{Float32}}}()
    net.params["W1"] = std * randn(Float32, input_size, hidden_size)
    net.params["b1"] = zeros(Float32, hidden_size)
    net.params["W2"] = std * randn(Float32, hidden_size, output_size)
    net.params["b2"] = zeros(Float32, output_size)

    net.layers = OrderedDict{String, Union{Affine, Relu, SoftmaxWithLoss}}()
    net.layers["Affine1"] = Affine(net.params["W1"], net.params["b1"])
    net.layers["Relu1"] = Relu()
    net.layers["Affine2"] = Affine(net.params["W2"], net.params["b2"])

    net.last_layer = SoftmaxWithLoss()

    return net
end

function predict(net::TwoLayerNet, x::Matrix{Float32})::Matrix{Float32}
    for (_, layer) in net.layers
        x = forward(layer, x)
    end
    return x
end

function loss(net::TwoLayerNet, x::Matrix{Float32}, t::Matrix{Float32})::Float32
    y = predict(net, x)
    return forward(net.last_layer, y, t)
end

function accuracy(
    net::TwoLayerNet,
    x::Matrix{Float32},
    t::Matrix{Float32},
)::Float32
    y = predict(net, x)
    y = vec(argmax(y, dims = 2))

    if ndims(t) != 1
        t = vec(argmax(t, dims = 2))
    end

    accuracy = sum(y .== t) / Float32(length(y))
    return accuracy
end

function gradient(net::TwoLayerNet, x, t)
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

    grads = Dict{String, Union{Matrix{Float32}, Vector{Float32}}}()
    grads["W1"] = net.layers["Affine1"].dW
    grads["b1"] = net.layers["Affine1"].db
    grads["W2"] = net.layers["Affine2"].dW
    grads["b2"] = net.layers["Affine2"].db

    return grads
end

function to_one_hot(t::Vector{Int}, num_classes::Int)::Matrix{Float32}
    one_hot = zeros(Float32, length(t), num_classes)
    for i in 1:length(t)
        one_hot[i, t[i] + 1] = 1
    end

    return one_hot
end

if abspath(PROGRAM_FILE) == @__FILE__
    const iters::Int = 10000
    const batch_size::Int = 100
    const learning_rate::Float32 = 0.10f0

    train_data = MLDatasets.MNIST(split = :train)
    test_data = MLDatasets.MNIST(split = :test)

    train_x, train_t = train_data.features, train_data.targets
    test_x, test_t = test_data.features, test_data.targets

    # 3次元配列を2次元配列に変換
    train_x =
        reshape(train_x, size(train_x, 3), size(train_x, 1) * size(train_x, 2))
    test_x = reshape(test_x, size(test_x, 3), size(test_x, 1) * size(test_x, 2))

    println("Size of train_x: ", size(train_x))
    println("Size of test_x: ", size(test_x))

    println("Size of train_t: ", size(train_t))
    println("Size of test_t: ", size(test_t))

    train_size = size(train_x, 2)
    loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = round(Int, train_size / batch_size)
    println("Iter per epoch: ", iter_per_epoch)

    net = TwoLayerNet(784, 50, 10)

    test_t = to_one_hot(test_t, 10)

    for i in 1:iters
        batch_mask = rand(1:train_size, batch_size)
        x_batch = train_x[batch_mask, :]
        t_batch = train_t[batch_mask]

        # t_batchをone-hot表現に変換
        t_batch = to_one_hot(t_batch, 10)

        # println("Size of x_batch: ", size(x_batch))
        # println("Size of t_batch: ", size(t_batch))

        net.layers["Affine1"].W = net.params["W1"]
        net.layers["Affine1"].b = net.params["b1"]
        net.layers["Affine2"].W = net.params["W2"]
        net.layers["Affine2"].b = net.params["b2"]

        grads = gradient(net, x_batch, t_batch)

        for (key, value) in net.params
            net.params[key] -= learning_rate * grads[key]
        end

        loss_value = loss(net, x_batch, t_batch)

        push!(loss_list, loss_value)

        if i % iter_per_epoch == 0
            train_acc_value = accuracy(net, x_batch, t_batch)
            push!(train_acc_list, train_acc_value)
            test_acc_value = accuracy(net, test_x, test_t)
            push!(test_acc_list, test_acc_value)
            println("Loss: ", loss_value)
            println("Train Accuracy: ", train_acc_value)
            println("Test Accuracy: ", test_acc_value)
        end
    end
end
