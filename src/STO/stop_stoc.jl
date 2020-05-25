function stop_stochastic_1(b, log_f::Function, x1::Vector, x2::Vector, it::Int64; verbose::Bool = false, nmax::Int64 = 500)
    m1 = zeros(2)
    m2 = zeros(2,2)
    
    n = 0
    for ind in b
        fXX = [log_f(ind, x1), log_f(ind, x2)]
        m1[:] += fXX
        m2[:,:] += fXX*fXX'
        n+=1
    end
    
    m1[:] /= n
    m2[:, :] /= n
    
    σ = sqrt(det(m2-m1))
    μ = m1[1] - m1[2]
    
    if verbose
        println("f(x_1) - f(x_2) ∼ N(", μ, ", ", σ^2/n, ") ")
    end
    
    if μ - 1.96*σ/sqrt(n) < 0 < μ + 1.96*σ/sqrt(n) #dont show it to M Bastin
        return true
    else
        return false
    end
    
    if it > nmax
        println("break by number of iterations")
        return true
    end
end
        
    