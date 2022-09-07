function update_Sigma(cur, dat, hyper)

	phistar = unique(cur["phi"], dims=2)
	mu = cur["mu"]

	nstar = size(phistar)[1]
	if size(phistar) == (2,1)
		phistar = reshape(phistar,:,1)
		nstar = 1
	end

	c1 = hyper["c"] + nstar 

	C1 = hyper["C"] 
	for i in 1:nstar
		#  logphi = log.(phistar[:,i:i]) # keep matrix type
		#  tmp_vec = logphi - mu
		tmp_vec = zeros(2) 
		tmp_vec[1] = log(phistar[1,i]) - mu[1]
		tmp_vec[2] = log(phistar[2,i]) - mu[2]

		C1 += tmp_vec * transpose(tmp_vec)
	end

	#  print(C1, "\n")
	if !ishermitian(C1)
		print(log.(phistar), "\n")
		print(c1, "\n")
		print(C1,"\n")
	end

	C1Inv = svd2inv(C1)
	SigmaInv = rand(Wishart(c1, C1Inv))
	Sigma = svd2inv(SigmaInv)
	#  Sigma = rand(InverseWishart(c1, C1), 1)[1]
	#  print(Sigma, "\n")

	return Sigma 
end
