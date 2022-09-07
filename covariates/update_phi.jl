function _sample_phi(M, theta, survival, nu, mu, Sigma)

	G0, D = get_G0_and_D(M, LogNormal(mu, sqrt(Sigma)), theta) 

	Omega_prop = zeros(M)
	Omega = zeros(M)
	for m in 1:M
		if nu == 1
			Omega_prop[m] = D[m] * pdf(Gamma(m,theta), survival)
		else
			Omega_prop[m] = D[m] * ccdf(Gamma(m,theta), survival)
		end
	end
	q0 = sum(Omega_prop)
	for m in 1:M
		Omega[m] = Omega_prop[m] / q0
	end
	cumOmega = cumsum(Omega)

	u1 = rand(Uniform(0,1),1)[1]
	m = findInterval(u1, cumOmega)
	utmp = u1*D[m] + G0[m]
	
	return quantile(LogNormal(mu, sqrt(Sigma)), utmp)
end

function update_phis(cur, dat, hyper)

	survivals = dat["survival"]
	nu = dat["nu"]
	x = dat["group"]
	Sigma = hyper["Sigma"]

	theta =  cur["theta"]
	alpha = cur["alpha"]
	M = cur["M"]
	mu = cur["mu"]

	n = length(survivals)

	phi_cur = cur["phi"] # 2 x n

	for i in 1:n
		xi = x[i]+1 # if x[i] = 0 then xi = 1, if x[i] = 1 then xi = 2
		_xi = 3 - xi # if x[i] = 0 then _xi = 2, if x[i] = 1 then _xi = 1

		M_i = M[xi]
		theta_i = theta[xi]

		G0, D = get_G0_and_D(M_i, LogNormal(mu[xi], sqrt(Sigma[xi,xi])), theta_i)

		q0 = 0.0 
		for m_i in 1:M_i
			if nu[i] == 1
				q0 += D[m_i] * pdf(Gamma(m_i,theta_i), survivals[i])
			else
				q0 += D[m_i] * ccdf(Gamma(m_i,theta_i), survivals[i])
			end
		end

		_phi_cur = phi_cur[:,1:end .!= i] # 2 x (n-1)
		qvec = zeros(n-1)
		intervals = [theta_i:theta_i:(M_i-1)*theta_i;]
		for j in 1:(n-1)
			mj = findInterval(_phi_cur[xi,j], intervals) 
			if nu[i] == 1
				qvec[j] = pdf(Gamma(mj, theta_i), survivals[i])
			else
				qvec[j] = ccdf(Gamma(mj, theta_i), survivals[i])
			end
		end

		u = rand(Uniform(0, alpha*q0 + sum(qvec)), 1)[1]
		
		if u < alpha*q0
			_phi_xi = rand(LogNormal(mu[_xi], sqrt(Sigma[_xi,_xi])), 1)[1]
			if _phi_xi <= 0 
				_phi_xi = 1.0e-10
			elseif _phi_xi == Inf
				_phi_xi = 1.0e10 
			end

			mu_cond = mu[xi] + Sigma[xi,_xi]/Sigma[_xi,_xi]*(_phi_xi-mu[_xi])
			Sigma_cond = Sigma[xi,xi] - Sigma[xi,_xi]*Sigma[_xi,xi]/Sigma[_xi,_xi]

			phi_xi = _sample_phi(M_i, theta_i, survivals[i], nu[i], mu_cond, Sigma_cond)
			if phi_xi <= 0 
				phi_xi = 1.0e-10
			elseif phi_xi == Inf
				phi_xi = 1.0e10 
			end

			phi_cur[xi,i] = phi_xi
			phi_cur[_xi,i] = _phi_xi

		else
			idx = sample([1:1:(n-1);], Weights(qvec))
			phi_cur[:,i] = _phi_cur[:,idx]
		end
	end

	return phi_cur
end
