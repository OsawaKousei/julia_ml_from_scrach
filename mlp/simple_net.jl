mutable struct SimpleNet
    W1::Matrix{Float32}
end

function SimpleNet()
    return SimpleNet(randn(Float32, 2, 3))
end

function predict(net::SimpleNet, x::Matrix{Float32})::Matrix{Float32}
    return x * net.W1
end

function loss(net::SimpleNet, x::Matrix{Float32}, t::Matrix{Float32})::Float32
    z = predict(net, x)
    y = softmax(z)
    return cross_entropy_error(y, t)
end

if abspath(PROGRAM_FILE) == @__FILE__
    net = SimpleNet()
    x = [0.6f0 0.9f0]
    t = [0.0f0 0.0f0 1.0f0]
    println(loss(net, x, t))
end