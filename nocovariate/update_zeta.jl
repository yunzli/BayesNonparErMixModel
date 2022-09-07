function update_zeta(cur::Dict, dat::Dict, hyper::Dict)

	a_zeta = hyper["a_zeta"]
	b_zeta = hyper["b_zeta"]
	phis = cur["phi"] 

	phis_star = unique(phis)
	n_star= length(phis_star)

	zeta = rand(InverseGamma(a_zeta+n_star, b_zeta+sum(phis_star)),1)[1]

	return zeta
end
