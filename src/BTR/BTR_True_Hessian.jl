
## ---------------------        btr         --------------------
function btr(f::Function, g!::Function, H!::Function, state::BTRState, x0::Vector, tTest::Function;
        verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64 = 1e-6, time_tol::Float64 = 120.0,
        accumulate!::Function, accumulator::Array)

    b = BTRDefaults() # init parametres par defaut
    state.fx = f(x0) # init f(x0)
    g!(x0, state.g)
    state.Δ = 0.1*norm(state.g)
    H!(x0, state.H)

    #function sparse_matrix(H::Matrix):
        #s = 0
        #for c in H
        #    if abs(c) < 1e-9
        #        s += 1
        #    end
        #end
        #println("$(100*s/length(H)) %")
    #end
    #sparse_matrix(state.H)

    state.step = x0
    state.ρ = 1.0
    state.start = time_ns()

    function model(s::Vector, g::Vector, H::Matrix)
        return dot(s, g)+0.5*dot(s, H*s)
    end

    while !Stop_optimize_mod(state, b, tTest, nmax = nmax, tol = epsilon, time_tol = time_tol)
        state.start = time_ns()
        accumulate!(state, accumulator)
        if verbose
            println(state.iter+1)
            #println(state)
        end
        state.step = TruncatedCG(state)
        state.xcand = state.x+state.step
        fcand = f(state.xcand)
        state.ρ = (fcand-state.fx)/(dot(state.step, state.g)+0.5*dot(state.step, state.H*state.step))
        if acceptCandidate!(state, b)
            state.x = copy(state.xcand)
            g!(state.x, state.g)
            H!(state.x, state.H)

            #sparse_matrix(state.H)

            state.fx = fcand
        end
        updateRadius!(state, b)
        state.iter += 1
        if verbose
            println("$((time_ns()-state.start)/1e9) s")
        end
    end
    return state, accumulator
end

#---------------------      OPTIM_btr_TH        -------------------------
# ----------- Fonction MAIN : optimisation Basic Trust Region True Hessian ----
function OPTIM_btr_TH(f::Function, g!::Function, H!::Function,
                x0::Vector; verbose::Bool = true,
                nmax::Int64 = 1000, epsilon::Float64 = 1e-4, time_tol::Float64 = 1e4, tTest::Function = par -> false)

    # -- Initialisation -----
    function accumulate!(state::BTRState, acc::Vector)
        push!(acc, state.fx)
    end
    accumulator = []

    # Definission de la matrice Hessienne, pour def du BTRState
    H = Array{Float64, 2}(I, length(x0), length(x0))
    state = BTRState(H)     # cf BTRBASE.jl
    #println(typeof(state))
    state.x = copy(x0)
    state.iter = 0
    state.g = zeros(length(x0))

    # ------ resolution par TR True Hessian ---- coeur de la fonction
    state, accumulator = btr(f, g!, H!, state, x0, tTest,
                verbose = verbose, nmax = nmax, epsilon = epsilon, time_tol = time_tol,
                accumulate! = accumulate!, accumulator = accumulator)

    # resultats de l'optimisation :
    #       -- state = etat final
    #       -- accumulator = ensemble des valeurs de f(x) visitees
    return state, accumulator
end
