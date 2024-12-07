include("gradient.jl")

mutable struct TestStruct
    x1::Float32
    x2::Float32
    var::Float32
end

function TestStruct(x1::Float32, x2::Float32)::TestStruct
    return TestStruct(x1, x2, 0.0f0)
end

function gradient_descent(
    f::Function,
    init_x::TestStruct,
    lr::Float32,
    step_num::Int,
)::TestStruct
    x = deepcopy(init_x)
    for i in 1:step_num
        grad = numerical_gradient(f, vec([x.x1, x.x2]))
        x.x1 -= lr * grad[1]
        x.x2 -= lr * grad[2]
        println("x: ", x.x1)
    end
    return x

end

if abspath(PROGRAM_FILE) == @__FILE__
    test_struct = TestStruct(3.0f0, 4.0f0)
    func = (x::Vector{Float32}) -> sum(x .^ 2)
    println(gradient_descent(func, test_struct, 0.1f0, 100))
end