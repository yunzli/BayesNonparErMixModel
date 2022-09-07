function update_alpha(cur, dat, hyper)

	n = length(dat["survival"])
	a_alpha = hyper["a_alpha"]
	b_alpha = hyper["b_alpha"]

	phis = cur["phi"]
	n_star = length(unique(phis))

	alpha = cur["alpha"]
	eta = rand(Beta(alpha+1, n),1)[1]

	weight = (a_alpha + n_star - 1) / (n * (1 / b_alpha - log(eta)) + a_alpha + n_star - 1) 

	u = rand(Uniform(0,1), 1)[1] 
	if u < weight
		alpha = rand(Gamma(a_alpha+n_star, 1 / (1 / b_alpha - log(eta))), 1)[1]
	else
		alpha = rand(Gamma(a_alpha+n_star-1, 1 / (1 / b_alpha - log(eta))), 1)[1]
	end

	return alpha 

end
