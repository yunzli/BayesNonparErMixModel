"""
    update_M(cur, dat, hyper)

Update the number of mixture components
"""
function update_M(cur, dat, hyper)

    survivals = dat["survival"]
    n = length(survivals)
    nu = dat["nu"]

    M1 = hyper["M1"]
    M2 = hyper["M2"]
    theta = cur["theta"]
    phis = cur["phi"]

    M_pool = [Int(ceil(M1 / theta)):1:Int(ceil(M2 / theta));]

    M_probs = zeros(length(M_pool))
    loglikelihoods = zeros(length(M_pool))

    for (i_m, M) in enumerate(M_pool)

        intervals = [theta:theta:(M-1)*theta;]

        for i in 1:n
            m = findInterval(phis[i], intervals)

            if nu[i] == 1
                loglikelihoods[i_m] += logpdf(Gamma(m, theta), survivals[i])
            else
                loglikelihoods[i_m] += logccdf(Gamma(m, theta), survivals[i])
            end
        end
    end

    max_ll = maximum(loglikelihoods)
    for (i_m, M) in enumerate(M_pool)
        M_probs[i_m] = exp(loglikelihoods[i_m] - max_ll)
    end

    return sample(M_pool, weights(M_probs))
end
