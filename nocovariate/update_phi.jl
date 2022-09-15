"""
    _update_phi(M, theta, zeta, alpha, phis, t, nu)

Update each latent variable ϕ 
"""
function _update_phi(M, theta, zeta, alpha, phis, t, nu)

    G0, D = get_G0_and_D(M, Exponential(zeta), theta)
    Omega_prop = zeros(M) 
    Omega = zeros(M) 
    intervals = [theta:theta:(M-1)*theta;]

    for m in 1:M
        if nu == 1
            Omega_prop[m] = D[m] * pdf(Gamma(m, theta), t) 
        else
            Omega_prop[m] = D[m] * ccdf(Gamma(m, theta), t) 
        end
    end
    q0 = sum(Omega_prop)
    for m in 1:M
        Omega[m] = Omega_prop[m] / q0
    end

    qcollect = Dict()
    for j in eachindex(phis)
        if haskey(qcollect, phis[j])
            qcollect[phis[j]] += 1
        else
            qcollect[phis[j]] = 1
        end
    end
    size = length(qcollect)

    _phi_star = zeros(size) 
    _n = zeros(size) 
    count = 1 
    for (key, value) in qcollect
        _phi_star[count] = key
        _n[count] = value
        count += 1 
    end

    qvec = zeros(size)
    for j in 1:size
        m = findInterval(_phi_star[j], intervals)
        if nu == 1
            qj = pdf(Gamma(m, theta), t) 
        else
            qj = ccdf(Gamma(m, theta), t) 
        end
        qvec[j] = _n[j] * qj
    end

    u = rand(Uniform(0, alpha * q0 + sum(qvec)), 1)[1]

    if u < alpha * q0
        u1 = rand(Uniform(0, 1), 1)[1]
        cumOmega = cumsum(Omega)
        m = findInterval(u1, cumOmega)
        utmp = u1 * D[m] + G0[m]
        phi_new = quantile(Exponential(zeta), utmp)
    else
        # idx = sample([1:1:size;], Weights(qvec))
        # phi_new = _phi_star[idx] 
        phi_new = sample(_phi_star, Weights(qvec))
    end

    return phi_new 
end

function update_phis(cur, dat, hyper)
    """
        update_phis(cur, dat, hyper)

    Update latent variables ϕᵢ for i = 1,..., n
    """
    nus = dat["nu"]
    survivals = dat["survival"]
    n = length(survivals)

    M = cur["M"]
    theta = cur["theta"]
    zeta = cur["zeta"]
    alpha = cur["alpha"]
    phi_cur = cur["phi"]

    for i in 1:n
        phi_cur[i] = _update_phi(M, theta, zeta, alpha, phi_cur[1:end.!=i], survivals[i], nus[i]) 
    end

    return phi_cur
end
