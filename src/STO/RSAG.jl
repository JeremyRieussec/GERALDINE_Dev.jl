# Retrospective Stochastic Agregated Gradient

function RSAG(x0::Vector, f::Function, g!::Function, batch_test, logf::Function, batch,
        alpha::Function, beta::Function, lambda::Function, k;
        verbose::Bool = false, nmax::Int64 = 1000)

    Xag = copy(x0) # X agregated
    Xmd = copy(x0) # X modifie avant de calcule le gradient
    Xk = copy(x0) # X iteration k
    grad = zeros(length(X0))
    g!(x_md, grad, batch_test)
    k = 0

    xprev = Xmd+grad
    while !stop_stochastic_1(batch_test, log_f, Xprev, Xmd, it, verbose = verbose, nmax = nmax)
        k += 1
        x_prev[:] = Xmd[:]
        Xmd[:] = (1-alpha(k))*Xag + alpha(k)Xk

        g!(Xmd, grad, batch)
        next!(batch)

        Xk[:] = Xk - lambda(k)*grad
        Xag = Xmd - beta(k)*grad
    end
    return Xk, Xmd, Xag
end

function OPTIM_AGRESSIVE_RSAG(f::Function, ∇f!::Function , batch, log_f::Function; x0::Vector, L::Float64, nmax::Int64 = 500,
        ϵ::Float64 = 1e-4, verbose::Bool = false, n_test::Int64 = 500, n_optim::Int64 = 100)

    batch_test = copy(batch)
    batch_test.n = n_test
    batch_optim = copy(batch_test)
    next!(batch_optim)
    batch_optim.n = n_optim
    define_start!(batch_optim)

    alpha = k::Int64 -> 2/(k+1)

    beta = k::Int64 -> 1/(2*lambda)

    lambda = k::Int64 -> k/(4*lambda)

    trash, Xmd, trash2 = RSAG(x0, f, ∇f!, batch_test, logf, batch_optim,
        alpha, beta, lambda, verbose = verbose, nmax = nmax)

    return Xmd
end
