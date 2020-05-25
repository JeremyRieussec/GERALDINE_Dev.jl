# Average Gradient

function AG(x0::Vector, f::Function, ∇f!::Function,
        α::Function, β::Function, γ::Function;
        nmax::Int64 = 500, ϵ::Float64 = 1e-4, verbose::Bool = false)
    x = x0
    x_ag = Array{Float64}(undef, length(x0))
    x_md = Array{Float64}(undef, length(x0))
    x_ag[:] = x0[:]
    ∇f_md = zeros(length(x0))
    for k in 1:nmax
        if verbose
            print(k, "  ")
            println("  ", x_md)
        end

        α_k = α(k)
        β_k = β(k)
        γ_k = γ(k)
        x_md = (1-α_k)*x_ag + α_k*x

        ∇f!(x_md, ∇f_md)

        if Stop_optimize(f(x_md), ∇f_md, k, nmax = nmax)
            println("break by norm at ", k)
            break
        end

        x -= γ_k*∇f_md
        x_ag = x_md - β_k*∇f_md
    end
    return x, x_md, x_ag
end

function OPTIM_AGRESSIVE_AG(f::Function, ∇f!::Function; x0::Vector, L::Float64, nmax::Int64 = 500,
        ϵ::Float64 = 1e-4, verbose::Bool = false)

    α = (k::Int64 -> 2/(k+1))
    β = (k::Int64 -> 1/(2*L))
    γ = (k::Int64 -> k/(4*L))
    trash, x, trash2 = AG(x0, f, ∇f!, α, β, γ;
            nmax = nmax, ϵ = ϵ, verbose = verbose)
    return x
end
