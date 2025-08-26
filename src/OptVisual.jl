module OptVisual
using ForwardDiff
using Plots
using LinearAlgebra

export Rosenbrock, Rastrigin, Spherefunction, Himmelblaufunction, L1norm, GD, NAG, optvisual, norm

abstract type AbstractTestFunction end


struct Rosenbrock <: AbstractTestFunction
    d::Int
end

struct Rastrigin <: AbstractTestFunction
    d::Int
    A::Float64   
end

struct Spherefunction <: AbstractTestFunction
    d::Int
end

struct Himmelblaufunction <: AbstractTestFunction
    d::Int
end

struct L1norm <: AbstractTestFunction
    d::Int
end

abstract type AbstractOptMethod end

struct GD <: AbstractOptMethod
    η::Float64      
    maxit::Int      
    log_every::Int  
end

struct NAG <: AbstractOptMethod
    η::Float64
    β::Float64
    maxit::Int
    v::Vector{Float64}
    log_every::Int
end

function Opt(fun::AbstractTestFunction, method::GD, x::AbstractVector)
    g = grad(fun, x)
    x -= method.η * g
    return x
end

function Opt(fun::AbstractTestFunction, method::NAG, x::AbstractVector)
    g = grad(fun, x .- method.β .* method.v)   
    method.v .= method.β .* method.v .+ method.η .* g
    return x .- method.v
end

#function
function f(fun::Rosenbrock, x::AbstractVector)
    s = 0.0
    for i in 1:fun.d-1
        s += (1 - x[i])^2 + 100 * (x[i+1] - x[i]^2)^2
    end
    return s
end

function f(fun::Rastrigin, x::AbstractVector)
    return fun.A*fun.d + sum(x.^2 .- fun.A * cos.(2π .* x))
end


function f(::Spherefunction, x::AbstractVector)
    return norm(x,2)^2
end

function f(::Himmelblaufunction, x::AbstractVector)
    return (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end

function f(::L1norm,x::AbstractVector)
    return norm(x,1)
end

function grad(fun::AbstractTestFunction, x0::AbstractVector)
    g = ForwardDiff.gradient(y -> f(fun, y), x0)
    return g
end

function optvisual(fun::AbstractTestFunction, opt::AbstractOptMethod; x0)
    x = copy(x0)
    fvals = Float64[]
    xs = Vector{Vector{Float64}}()

    for i in 1:opt.maxit
        x = Opt(fun, opt, x)
        if i % opt.log_every == 0
            push!(fvals, f(fun, x))
            push!(xs, copy(x))
        end
    end

    iters = 1:opt.log_every:opt.maxit
    p1 = plot(iters, fvals; yscale=:log10, lw=2, label=string(typeof(opt)), title="convergence curve")
    
    p2 = nothing
    if length(x0) == 2
        xs1 = range(-5, 5, length=200)
        xs2 = range(-5, 5, length=200)
        Z = [f(fun, [x,y]) for y in xs2, x in xs1]
    
        p2 = contour(xs1, xs2, Z; levels=50, linewidth=0.5, title="optimization trajectory")
        plot!(p2, first.(xs), last.(xs), marker=:circle, label="trajectory")
    end
    
    if p2 === nothing
        display(p1)
    else
        display(plot(p1, p2, layout=(1,2), size=(900,400)))
    end
    
    
end

end