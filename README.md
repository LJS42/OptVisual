# OptVisual.jl

Learning Julia right now...

**OptVisual.jl** is a lightweight package I wrote while learning **Julia**,  
for visualizing simple optimization algorithms.

It provides:
- A set of standard benchmark functions (Sphere, Rosenbrock, Rastrigin, Himmelblau, L1 norm)
- Basic first-order optimization methods (Gradient Descent, Nesterov Accelerated Gradient)
- Visualization of **convergence curves** and **2D optimization trajectories** on contour plots

---

## Installation

This package is not yet in the Julia General registry — install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/LJS42/OptVisual.jl.git")
```

## Example
```julia
using OptVisual

# Define a 2D test function (Himmelblau)
fun = Himmelblaufunction(2)

# Define an optimization method (Gradient Descent)
method = GD(0.01, 200, 10)

# Visualize optimization process
optvisual(fun, method; x0=[-3.0, -3.0])
```

## Supports self-defined functions and optimization methods

Function:
```julia

struct yourfunction <: AbstractTestFunction
    d::Int #dimension
end

function f(::yourfunction, x::AbstractVector)
    #your function here
end
```

Method:
```julia
struct yourmethod <: AbstractOptMethod
    #your hyperpameter
end

function Opt(fun::AbstractTestFunction, method::yourmethod, x::AbstractVector)
    #your method
end
