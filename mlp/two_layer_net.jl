using LinearAlgebra
using MLDatasets
using DataStructures
using PyPlot

include("activation_function.jl")
include("loss_function.jl")
include("layer.jl")
include("gradient.jl")

mutable struct TwoLayerNet{T <: Real}
    layers::OrderedDict{String, Union{Affine{T}, Relu{T}}}
    last_layer::SoftmaxWithLoss{T}

    function TwoLayerNet{T}(
        input_size::Int,
        hidden_size::Int,
        output_size::Int,
        std = 0.01,
    ) where T <: Real
        std = convert(T, std)
        layers = OrderedDict{String, Union{Affine{T}, Relu{T}}}()
        layers["Affine1"] = Affine{T}(std * randn(T, input_size, hidden_size), zeros(T, hidden_size))
        layers["Relu1"] = Relu{T}()
        layers["Affine2"] = Affine{T}(std * randn(T, hidden_size, output_size), zeros(T, output_size))

        last_layer = SoftmaxWithLoss{T}()

        return new{T}(layers, last_layer)
    end
end

function predict(net::TwoLayerNet, x::Matrix{T})::Matrix{T} where T <: Real
    for (key, layer) in net.layers
        x = forward(layer, x)
    end
    return x
end

function loss(net::TwoLayerNet, x::Matrix{T}, t::Matrix{T})::T where T <: Real
    y = predict(net, x)
    return forward(net.last_layer, y, t)
end

function accuracy(
    net::TwoLayerNet,
    x::Matrix{T},
    t::Matrix{T},
)::T where T <: Real
    y = predict(net, x)
    y = vec(argmax(y, dims = 2))

    t = vec(argmax(t, dims = 2))


    accuracy = sum(y .== t) / length(y)
    return accuracy
end

# function numerical_gradient(
#     net::TwoLayerNet,
#     x::Matrix{Float64},
#     t::Matrix{Float64},
# )::Dict{String, Union{Matrix{Float64}, Vector{Float64}}}

#     grads = Dict{String, Union{Matrix{Float64}, Vector{Float64}}}()

#     func_W1(W1)::Float64 = loss(
#         TwoLayerNet(W1, net.params["W2"], net.params["b1"], net.params["b2"]),
#         x,
#         t,
#     )
#     grads["W1"] = numerical_gradient(func_W1, net.params["W1"])
#     func_b1(b1)::Float64 = loss(
#         TwoLayerNet(net.params["W1"], net.params["W2"], b1, net.params["b2"]),
#         x,
#         t,
#     )
#     grads["b1"] = numerical_gradient(func_b1, net.params["b1"])
#     func_W2(W2)::Float64 = loss(
#         TwoLayerNet(net.params["W1"], W2, net.params["b1"], net.params["b2"]),
#         x,
#         t,
#     )
#     grads["W2"] = numerical_gradient(func_W2, net.params["W2"])
#     func_b2(b2)::Float64 = loss(
#         TwoLayerNet(net.params["W1"], net.params["W2"], net.params["b1"], b2),
#         x,
#         t,
#     )
#     grads["b2"] = numerical_gradient(func_b2, net.params["b2"])

#     return grads
# end

function gradient(
    net::TwoLayerNet,
    x::Matrix{T},
    t::Matrix{T},
)::Dict{String, Union{Matrix{T}, Vector{T}}} where T <: Real
    # forward
    loss(net, x, t)

    # backward
    d_out = convert(T, 1.0) # lossはlast_layerに格納されているが、形式的に1を代入
    d_out = backward(net.last_layer, d_out)

    layers = collect(net.layers)
    layers = reverse(layers)
    for (key, layer) in layers
        d_out = backward(layer, d_out)
    end

    grads = Dict{String, Union{Matrix{T}, Vector{T}}}()
    grads["W1"] = net.layers["Affine1"].dW
    grads["b1"] = net.layers["Affine1"].db
    grads["W2"] = net.layers["Affine2"].dW
    grads["b2"] = net.layers["Affine2"].db

    return grads
end

function update_params(
    net::TwoLayerNet,
    grads::Dict{String, Union{Matrix{T}, Vector{T}}},
    learning_rate::Float64,
) where T <: Real
    net.layers["Affine1"].W -= learning_rate * grads["W1"]
    net.layers["Affine1"].b -= learning_rate * grads["b1"]
    net.layers["Affine2"].W -= learning_rate * grads["W2"]
    net.layers["Affine2"].b -= learning_rate * grads["b2"]
end

function to_one_hot(t::Vector{Int}, num_classes::Int)::Matrix
    one_hot = zeros(Float64, length(t), num_classes)
    for i in 1:length(t)
        one_hot[i, t[i] + 1] = 1
    end

    return one_hot
end

if abspath(PROGRAM_FILE) == @__FILE__
    const iters::Int = 10000
    const batch_size::Int = 100
    const learning_rate::Float64 = 0.10

    train_data = MLDatasets.MNIST(split = :train)
    test_data = MLDatasets.MNIST(split = :test)

    train_x, train_t = Float64.(train_data.features), train_data.targets
    test_x, test_t = Float64.(test_data.features), test_data.targets

    println("Size of train_x: ", size(train_x))
    println("Size of test_x: ", size(test_x))

    # 3次元配列を2次元配列に変換
    train_x =
        reshape(train_x, size(train_x, 1) * size(train_x, 2), size(train_x, 3))
    test_x = reshape(test_x, size(test_x, 1) * size(test_x, 2), size(test_x, 3))

    # 1列目と2列目を入れ替え
    train_x = Matrix(train_x')
    test_x = Matrix(test_x')

    println("Size of train_x: ", size(train_x))
    println("Size of test_x: ", size(test_x))
    println("type of train_x: ", typeof(train_x))
    println("type of test_x: ", typeof(test_x))

    println("Size of train_t: ", size(train_t))
    println("Size of test_t: ", size(test_t))

    train_size = size(train_x, 1)
    println("Train size: ", train_size)
    loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = round(Int, train_size / batch_size)
    println("Iter per epoch: ", iter_per_epoch)

    net = TwoLayerNet{Float64}(784, 50, 10, 0.01f0)

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

        # バッチの最初のデータを画像に変換し、ラベルを表示
        # img = reshape(x_batch[1, :], 28, 28)
        # imshow(img, cmap = "gray")
        # println("Label: ", argmax(t_batch[1, :]) - 1)

        # grads = numerical_gradient(net, x_batch, t_batch)
        grads = gradient(net, x_batch, t_batch)

        # 勾配が正しいか確認し、数値微分と誤差が小さければOK
        # for (key, value) in grads
        #     println("Key: ", key)
        #     println("Diff: ", maximum(abs.(numerical_grads[key] .- value)))
        # end

        update_params(net, grads, learning_rate)

        loss_value = loss(net, x_batch, t_batch)

        push!(loss_list, loss_value)

        if i % iter_per_epoch == 0
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
