# Aggregated Gradient

function AG(f::Function, ∇f!::Function,
        α::Function, β::Function, γ::Function, state::AGState;
        nmax::Int64 = 500, epsilon::Float64 = 1e-4, verbose::Bool = false, accumulate!::Function = (x,y) -> nothing, acc = [])

    while !Stop_optimize(state.fx_md , state.∇f_md, state.it, nmax = nmax, tol = epsilon)
        state.it += 1
        accumulate!(state, acc)
        if verbose
            println(state)
        end

        α_k = α(state.it)
        β_k = β(state.it)
        γ_k = γ(state.it)
        state.x_md = (1-α_k)*state.x_ag + α_k*state.x

        ∇f!(state.x_md, state.∇f_md)
        state.fx_md = f(x_md)

        state.x -= γ_k*state.∇f_md
        state.x_ag = state.x_md - β_k*state.∇f_md
    end
    return state, acc
end

function OPTIM_AGRESSIVE_AG(f::Function, ∇f!::Function; x0::Vector, L::Float64, nmax::Int64 = 500,
        epsilon::Float64 = 1e-4, verbose::Bool = false, acc!::Function = (sta, acc) -> nothing, acc = [])

    α = (k::Int64 -> 2/(k+1))
    β = (k::Int64 -> 1/(2*L))
    γ = (k::Int64 -> k/(4*L))

    state = AGState(x0)

    state, acc = AG(f::Function, ∇f!::Function, α, β, γ, state;
        nmax = nmax, epsilon = epsilon, verbose = verbose, acc! = acc!, acc = acc)

    return state.x_md
end
