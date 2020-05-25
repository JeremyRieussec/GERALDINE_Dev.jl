mutable struct AGState <: AbstractState
    x::Vector
    x_ag::Vector
    x_md::Vector
    ∇f_md::Vector
    fx_md::Float64
    it::Int64
    function AGState(x0::Vector)
        n = new()
        n.x = copy(x0)
        n.x_ag = copy(x0)
        n.x_md = copy(x0)
        n.∇f_md = zeros(length(x0))
        n.fx_md = 0.0
        n.it = 0
        return n
    end
end

import Base.println

function println(a::AGState)
    println("Iteration", a.it, ", fx = ", a.fx_md, "||∇f(x)|| = ", norm(n.∇f_md))
    
    println(round.(a.x_md, digits = 3))
end
    