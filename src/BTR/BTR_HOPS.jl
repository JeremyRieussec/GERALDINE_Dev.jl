# RAPPEL : le point d'exclamation apres nom fonction (ex : f!) est pour dire que
#          la fonction modifie les arguments

# f : fonction a optimiser
# g! : gradient
# Hx : fonction de calcul de produit Hessien-vecteur
# state : etat courant
# x0 : vecteur de depart
# epsilon : critere d'arret --> condition sur le gradient ici
#                               A MODIFIER : (espsilon_g , epsilon_H)-optimalite
# verbose : Bool pour activer Tracage
# accumulate! : fonction pour mettre a jour tracage de l'ensemble des etats parcourus
# accumulator : etats parcourus

#--------------        btr_HOPS        ------------------------
function btr_HOPS(f::Function, g!::Function, Hx::Function, state::BTRState, x0::Vector, tTest::Function;
        verbose::Bool = false, nmax::Int64 = 1000, epsilon::Float64 = 1e-6, time_tol::Float64 = 120.0,
        accumulate!::Function, accumulator::Array)
    # ---- initialisation ------
    b = BTRDefaults() # cf BTRBASE.jl : init parametres TR
    state.fx = f(x0)

    if verbose  # Pour affichage du tracage des valeurs de f(x)
        println(state.fx)
    end

    g!(x0, state.g, state.H)

    state.Δ = 0.1*norm(state.g) # init du rayon de region de confiance
    state.step = x0
    state.ρ = 1.0 # init facteur precision modele
    state.start = time_ns() # init temps de depart

    # -------- Boucle optimisation ------------
    # Pour Stop_optimize_mod : cf GERALDINE/STO/stop_stoc.jl
    while !Stop_optimize_mod(state, b, tTest, nmax = nmax, tol = epsilon, time_tol = time_tol)
        state.start = time_ns()
        accumulate!(state, accumulator)

        if verbose # tracking
            println(state.iter+1)
            #println(state)
        end
        state.step = TCG_HOPS(state, Hx) # cf BTRBASE.jl : Truncated CG with HOPS
        state.xcand = state.x+state.step # candidat potentiel --> a valider avec rho
        fcand = f(state.xcand)
        state.ρ = (fcand-state.fx)/(dot(state.step, state.g)+0.5*dot(state.step, Hx(state.step)))
        if acceptCandidate!(state, b) # cf BTRBASE.jl
            state.x = copy(state.xcand)
            g!(state.x, state.g, state.H)
            state.fx = fcand
        end
        updateRadius!(state, b) # cf BTRBASE.jl
        state.iter += 1
        if verbose
            println(state.fx)
            println("$((time_ns()-state.start)/1e9) s")
        end
    end
    return state, accumulator
end

#------------------     OPTIM_btr_HOPS      ---------------------------------
# f : fcontion a optimiser
# g_score! : score de la fonction objectif
# x0 : vecteur initial
# weights : vecteur nombre parametres par individus, qui peuvent etre
#               -- features par individus pour regression logistique lineaire
#               -- weight + bias par couche pour MultiLayer Perceptron
#               -- ...
# verbose : tracage
# nmax : nombre iterations maximal
# epsilon : precision critere arret
# tTest : critere arret base sur tTest ---> cf AMLET/Tests/MNIST.jl
#

# Fonction MAIN pour Optimisation avec BTR HOPS
function OPTIM_btr_HOPS(f::Function, g_score!::Function, x0::Vector, weights::Vector;
        verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64 = 1e-4,
        time_tol::Float64 = 1e4, tTest::Function = par -> false)

    # ACCUMULATE : stockage de l'ensemble des valeurs de f(x) parcourues
    function accumulate!(state::BTRState, acc::Vector)
        push!(acc, state.fx)
    end
    accumulator = []

    n = sum(weights)
    inds = length(weights)
    S = [zeros(length(x0)) for i = 1:inds] # Matrice des scores
    state = BTRState(S) # cf BTRBASE.jl
    state.x = copy(x0)
    state.iter = 0
    state.g = zeros(length(x0))

# !!! Definissions du produit HESSIEN-VECTEUR
    Hx(x::Vector) = (1/n)*sum(weights[i]*dot(state.H[i], x)*state.H[i] for i in 1:inds)

# Resolution par TR HOPS --------- coeur de la fonction
    state, accumulator = btr_HOPS(f, g_score!, Hx, state, x0, tTest,
                verbose = verbose, nmax = nmax, epsilon = epsilon, time_tol = time_tol,
                accumulate! = accumulate!, accumulator = accumulator)

    # resultats de l'optimisation :
    #       -- state = etat final
    #       -- accumulator = ensemble des valeurs de f(x) visitees
    return state, accumulator
end
