function _update_M(theta, phi, nu, survival, M1, M2)
	"""
	This function samples a marginal M from its posterior distribution
	"""
	n = length(nu)

	M_pool = [Int(ceil(M1/theta)):1:Int(ceil(M2/theta));]
	M_probs = zeros(length(M_pool))
	loglikelihoods = zeros(length(M_pool))
for (i_m, M) in enumerate(M_pool)

		intervals = collect(theta:theta:(M-1)*theta)

		for i in 1:n
			m = findInterval(phi[i], intervals) 

			if nu[i] == 1
				loglikelihoods[i_m] += logpdf(Gamma(m, theta), survival[i])
			else
				loglikelihoods[i_m] += logccdf(Gamma(m, theta), survival[i])
			end
		end
	end

	max_ll = maximum(loglikelihoods)
	for (i_m, M) in enumerate(M_pool)
		M_probs[i_m] = exp(loglikelihoods[i_m] - max_ll)
	end

	return sample(M_pool, Weights(M_probs))
end

function update_M(cur, dat, hyper)

	survivalC = dat["survivalC"]
	survivalT = dat["survivalT"]
	nuC = dat["nuC"]
	nuT = dat["nuT"]
	indexC = dat["indexC"]
	indexT = dat["indexT"]
	phi = cur["phi"]
	phiC = phi[1,indexC]
	phiT = phi[2,indexT]
	theta = cur["theta"]
	thetaC = theta[1]
	thetaT = theta[2] 

	MC1 = hyper["M_C1"]
	MC2 = hyper["M_C2"]
	MT1 = hyper["M_T1"]
	MT2 = hyper["M_T2"]

	MC = _update_M(thetaC, phiC, nuC, survivalC, MC1, MC2)
	MT = _update_M(thetaT, phiT, nuT, survivalT, MT1, MT2)

	return [MC,MT] 
end
