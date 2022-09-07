function update_phis(cur, dat, hyper)

	survivals = dat["survival"]
	n = length(dat["survival"])
	nu = dat["nu"]

	alpha = cur["alpha"]
    theta = cur["theta"]
	zeta = cur["zeta"]
	M = cur["M"] 
	
	G0, D = get_G0_and_D(M, Exponential(zeta), theta) 
	
	Omega_prop = zeros(M) # Vector{Float64}(undef, M)
	Omega = zeros(M) # Vector{Float64}(undef, M)

	phi_cur = cur["phi"]
	intervals = [theta:theta:(M-1)*theta;]

	for i in 1:n

		for m in 1:M
			if nu[i] == 1 
				Omega_prop[m] = D[m] * pdf(Gamma(m,theta), survivals[i])
			else
				Omega_prop[m] = D[m] * ccdf(Gamma(m,theta), survivals[i])
			end
		end
		q0 = sum(Omega_prop) 
		for m in 1:M
			Omega[m] = Omega_prop[m] / q0 
		end 

		_phi_cur = phi_cur[1:end .!= i]
		qvec = zeros(n-1)
		for j in 1:(n-1)
		    mj = findInterval(_phi_cur[j], intervals)
			if nu[i] == 1
				qvec[j] = pdf(Gamma(mj, theta), survivals[i])
			else
				qvec[j] = ccdf(Gamma(mj, theta), survivals[i])
			end
		end

		u = rand(Uniform(0, alpha*q0 + sum(qvec)), 1)[1] 

		if u < alpha*q0
			u1 = rand(Uniform(0,1),1)[1]
			cumOmega = cumsum(Omega)
			m = findInterval(u1, cumOmega)
			utmp = u1*D[m] + G0[m]
			phi_cur[i] = quantile(Exponential(zeta), utmp)
		else
			idx = sample([1:1:(n-1);], Weights(qvec))
			phi_cur[i] = _phi_cur[idx]
		end
			


		#  table = countmap(phi_cur[1:end .!= i])
		#  _phi_star = collect(keys(table))
		#  _n = collect(values(table))
		#  size = length(table)
        #
		#  probs = zeros(size+1)
		#  probs[1] = alpha * q0
		#  for j in 1:size
		#      m = findInterval(_phi_star[j], collect(theta:theta:(M-1)*theta))
		#      if nu[i] == 1
		#          qj = pdf(Gamma(m, theta), survivals[i])
		#      else
		#          qj = ccdf(Gamma(m, theta), survivals[i])
		#      end
		#      probs[j+1] = _n[j] * qj
		#  end
		#
		#  A = sum(probs)
		#  cdfs = cumsum(probs)
        #
		#  u = rand(Uniform(0,1),1)[1]
		#  idx = findInterval(A*u, cdfs)
		#
		#  if idx != 1
		#      phi_cur[i] = _phi_star[idx-1]
		#  else
		#      u1 = rand(Uniform(0,1),1)[1]
		#      cumOmega = cumsum(Omega)
        #
		#      m = findInterval(u1, cumOmega)
        #
		#      if m != M
		#          tmp = rand(truncated(Exponential(zeta), lower=(m-1)*theta, upper=m*theta),1)[1]
		#      else
		#          tmp = rand(truncated(Exponential(zeta), lower=(M-1)*theta), 1)[1]
		#      end
		#      phi_cur[i] = tmp
		#  end
	end

	return phi_cur 
end 
