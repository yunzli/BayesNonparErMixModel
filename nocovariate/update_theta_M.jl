# update θ and M jointly 

function update_theta_M(cur, dat, hyper, l_rate)

	survivals = dat["survival"] 
	nu = dat["nu"] 

	n = length(survivals)

	a_theta = hyper["a_theta"]
	b_theta = hyper["b_theta"]
	M1 = hyper["M1"]
	M2 = hyper["M2"]

	phis = cur["phi"]
	M_cur = cur["M"]
	
	theta_cur = cur["theta"]
	sig = rand(Normal(0, exp(l_rate)),1)[1]
    theta_pro = exp(log(theta_cur) + sig)

	M_pool_cur = [ceil(Int32, M1/theta_cur):1:ceil(Int32, M2/theta_cur);]
	M_pool_pro = [ceil(Int32, M1/theta_pro):1:ceil(Int32, M2/theta_pro);]

	M_probs_pro = zeros(length(M_pool_pro))
	for (i, m) in enumerate(M_pool_pro)
		M_probs_pro[i] = 1/((M_cur-m)^2+1)
	end
	M_pro = sample(M_pool_pro, weights(M_probs_pro))

	M_probs_cur = zeros(length(M_pool_cur))
	for (i, m) in enumerate(M_pool_cur)
		M_probs_cur[i] = 1/((M_pro-m)^2+1)
	end

	# TODO 
	# check math for acceptence rate calculation 
	log_cur = log(theta_cur) + logpdf(Gamma(a_theta, b_theta), theta_cur) + log(sum(M_probs_cur)) 
	log_pro = log(theta_pro) + logpdf(Gamma(a_theta, b_theta), theta_pro) + log(sum(M_probs_pro))

	for i in 1:n
		m_cur = findInterval(phis[i], [theta_cur:theta_cur:(M_cur-1)*theta_cur;])
		m_pro = findInterval(phis[i], [theta_pro:theta_pro:(M_pro-1)*theta_pro;])

		if nu[i] == 1
			log_cur += logpdf(Gamma(m_cur, theta_cur), survivals[i])
			log_pro += logpdf(Gamma(m_pro, theta_pro), survivals[i])
		else
			log_cur += logccdf(Gamma(m_cur, theta_cur), survivals[i])
			log_pro += logccdf(Gamma(m_pro, theta_pro), survivals[i])
		end

	end

	if log(rand(Uniform(0,1))) < log_pro - log_cur
		M_cur = M_pro
		theta_cur = theta_pro
	end

	return [theta_cur, M_cur] 
end 
