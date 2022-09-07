function update_theta(cur::Dict, dat::Dict, hyper::Dict, l_rate::Float64)::Dict

	survivals = dat["survival"]
	n = length(survivals)
	nu = dat["nu"]

	a_theta = hyper["a_theta"]
	b_theta = hyper["b_theta"]

	phis = cur["phi"]
	M = cur["M"]

	theta_cur = cur["theta"]
	sig = rand(Normal(0, exp(l_rate)),1)[1]
	theta_pro = exp(log(theta_cur) + sig)

	log_cur_theta = log(theta_cur) + logpdf(Gamma(a_theta, b_theta), theta_cur)
	log_pro_theta = log(theta_pro) + logpdf(Gamma(a_theta, b_theta), theta_pro)

	for i in 1:n
		m_cur = findInterval(phis[i], [theta_cur:theta_cur:(M-1)*theta_cur;])
		m_pro = findInterval(phis[i], [theta_pro:theta_pro:(M-1)*theta_pro;])

		if nu[i] == 1 # observed 
			log_cur_theta += logpdf(Gamma(m_cur, theta_cur), survivals[i])
			log_pro_theta += logpdf(Gamma(m_pro, theta_pro), survivals[i])
		else # cencored 
			log_cur_theta += logccdf(Gamma(m_cur, theta_cur), survivals[i])
			log_pro_theta += logccdf(Gamma(m_pro, theta_pro), survivals[i])
		end
	end
	
	acc = 0 
	if log(rand(Uniform(0,1), 1)[1]) < log_pro_theta - log_cur_theta
		theta_cur = theta_pro
		acc = 1
	end
	
	res = Dict("theta" => theta_cur,
			   "acc"   => acc)

	return res 
end 
