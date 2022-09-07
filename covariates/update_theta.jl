function _update_theta(theta, M, a, b, survivals, nu, phi)
	"""
    This function calculates the marginal log-posterior function for theta
	"""
    
	n = length(nu)
	ll = log(theta) + logpdf(Gamma(a, b), theta)

	intervals = [theta:theta:(M-1)*theta;]
	for i in 1:n
		m = findInterval(phi[i], intervals)

		if nu[i] == 1
			ll += logpdf(Gamma(m, theta), survivals[i])
		else
			ll += logccdf(Gamma(m, theta), survivals[i])
		end
	end

	return ll 
end


function update_theta(cur, dat, hyper, Sig) 
	"""
	Update (thetaC, thetaT) using adaptive Metropolis algorithm 
	"""

	survivalC = dat["survivalC"]
	survivalT = dat["survivalT"]
	nuC = dat["nuC"]
	nuT = dat["nuT"]
	nC = length(nuC)
	nT = length(nuT) 
	indexC = dat["indexC"]
	indexT = dat["indexT"]

	phi = cur["phi"]
	phiC = phi[1,indexC]
	phiT = phi[2,indexT]

	M = cur["M"]
	MC = M[1]
	MT = M[2]

	theta_cur = cur["theta"]

	u = rand(Uniform(0,1),1)[1]
	if u < 0.95
		theta_pro = exp.(rand(MvNormal(log.(theta_cur), 2.38^2/2*Sig)))
	else
		theta_pro = exp.(rand(MvNormal(log.(theta_cur), 0.1^2/2*Diagonal(ones(2)))))
	end # adaptive MCMC, proposing a new candidate 

	log_curC = _update_theta(theta_cur[1], 
							 MC, 
							 hyper["a_C"], 
							 hyper["b_C"], 
							 survivalC, 
							 nuC, 
							 phiC) # thetaC at t 

	log_curT = _update_theta(theta_cur[2], 
							 MT, 
							 hyper["a_T"], 
							 hyper["b_T"], 
							 survivalT, 
							 nuT, 
							 phiT) # thetaT at t 

	log_proC = _update_theta(theta_pro[1], 
							 MC, 
							 hyper["a_C"], 
							 hyper["b_C"], 
							 survivalC, 
							 nuC, 
							 phiC) # thetaC * 

	log_proT = _update_theta(theta_pro[2], 
							 MT, 
							 hyper["a_T"], 
							 hyper["b_T"], 
							 survivalT, 
							 nuT, 
							 phiT) # thetaT * 

	log_cur = log_curC + log_curT
	log_pro = log_proC + log_proT 

	if log(rand(Uniform(0,1), 1)[1]) < log_pro - log_cur
		theta_cur = theta_pro
	end

	return theta_cur
end
