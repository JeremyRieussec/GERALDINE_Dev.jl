
#-------------------          BasicTrustRegion      -----------------------

struct BasicTrustRegion{T<:Real}
    η1::T  # Limite pour refus du pas
    η2::T # Acceptation du pas
    γ1::T # contraction rayon
    γ2::T # Extension rayon region de confiance
end

# constructeur Basic Trust Region
function BTRDefaults()
    return BasicTrustRegion(0.01, 0.9, 0.5, 0.5)
end

##-----------       Structure Etat de Basic Trust region        --------------

mutable struct BTRState{T} <: AbstractState where T
    # Declacration des attributs :
    iter::Int64 # numero iteration
    start::Int64 # point depart
    x::Vector # vecteur parametres
    xcand::Vector # x Candidate
    g::Vector # Gradient
    H::T # Hesienne
    step::Vector # pas pour mise a jour dans TR
    Δ::Float64 # Rayon region de confiance
    ρ::Float64 # facteur precision modele
    fx::Float64 # f(x)

    # constructeur avec type de Hessienne :
    #       -- True Hessienne
    #       -- BHHH --> stockage des scores
    #       -- BFGS
    #       -- L-BFGS --> Stockage des (s_i, y_i)
    function BTRState(H::T) where T
        state = new{T}()
        state.H = H
        return state
    end
end

import Base.println

##  ---------------     Fonction Print ------------------------------

function println(state::BTRState)
    println(round.(state.x, digits = 3))
end

#------------       acceptCandidate          ---------------
# Evaluation de la qualite du trial step
function acceptCandidate!(state::BTRState, b::BasicTrustRegion)
    if state.ρ >= b.η1
        return true
    else
        return false
    end
end

## --------------------     updateRadius        -----------------------
# Mise a jour du rayon de region de confiance
function updateRadius!(state::BTRState, b::BasicTrustRegion)
    if state.ρ >= b.η2
        stepnorm = norm(state.step)
        state.Δ = min(10e12, max(4*stepnorm, state.Δ))
    elseif state.ρ >= b.η1
        state.Δ *= b.γ2
    else
        state.Δ *= b.γ1
    end
end

## -------------------      stopCG          -------------------------
# critere arret de CG :
# -- max CG
# -- forcing term : ici sqrt(normg0) --> CV lin de Quasi-Newton
# A RAJOUTER :
# -- forcing sequence quelconque
# -- forcing term dynamique : cf Kejic
# -- forcing term Variance
function stopCG(normg::Float64, normg0::Float64, k::Int, kmax::Int)
    χ::Float64 = 0.1
    θ::Float64 = 0.5
    if (k == kmax) || (normg <= normg0*min(χ, normg0^θ))
        return true
    else
        return false
    end
end

## -----------------        TruncatedCG         -----------------------
# CG tronque (cf Steihaug)
# A MODIFIER :  prise en compte direction de suffisante courbure negative
# ---> (epsilon_g , epsilon_H)-optimalite
function TruncatedCG(state::BTRState) #  prend en argument etat courant BTR
    H = state.H # Hesienne
    g = state.g # Gradient
    Δ = state.Δ*state.Δ # rayon region confiance
    n = length(g) # dimemsion vecteur de parametres a optimiser
    s = zeros(n) # initialisation du vecteur de depart au VECTEUR NUL
    normg0 = norm(g)
    v = g # v re presente le residu a la solution, soit le gradient a l'iteration courante
    d = -v # d sont les directions conjuguees, d_0 = -g
    gv = dot(g, v)
    norm2d = gv
    norm2s = 0
    sMd = 0
    k = 0
    while ! stopCG(norm(g), normg0, k, n)
        Hd = H*d  # produit Hessienne vecteur
        κ = dot(d, Hd) # H-norm
        if κ <= 0 # Recherche direction courbure negative
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d # facteur pour
                                                                 #etre a la frontiere
            s += σ*d
            break
        end
        α = gv/κ
        norm2s += α*(2*sMd+α*norm2d)
        if norm2s >= Δ
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d
            s += σ*d
            break
        end
        s += α*d
        g += α*Hd
        v = g
        newgv = dot(g, v)
        β = newgv/gv
        gv = newgv
        d = -v+β*d
        sMd = β*(sMd+α*norm2d)
        norm2d = gv+β*β*norm2d
        k += 1
    end
    return s
end

## ---------------------------      TCG_HOPS        ---------------------------
# Truncated CG avec HOPS (= Hessian-free Outer Product of the Scores)
function TCG_HOPS(state::BTRState, Hx::Function)
    g = state.g
    Δ = state.Δ*state.Δ
    n = length(g)
    s = zeros(n)
    normg0 = norm(g)
    v = g
    d = -v
    gv = dot(g, v)
    norm2d = gv
    norm2s = 0
    sMd = 0
    k = 0
    while ! stopCG(norm(g), normg0, k, n)
        Hd = Hx(d) # c'est ici que tout se passe !!!!
        κ = dot(d, Hd)
        if κ <= 0
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d
            s += σ*d
            break
        end
        α = gv/κ
        norm2s += α*(2*sMd+α*norm2d)
        if norm2s >= Δ
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d
            s += σ*d
            break
        end
        s += α*d
        g += α*Hd
        v = g
        newgv = dot(g, v)
        β = newgv/gv
        gv = newgv
        d = -v+β*d
        sMd = β*(sMd+α*norm2d)
        norm2d = gv+β*β*norm2d
        k += 1
    end
    return s
end
