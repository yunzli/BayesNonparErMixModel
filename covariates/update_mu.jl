function update_mu(cur, dat, hyper)

	Sigma0Inv = hyper["Sigma0Inv"]
	SigmaInv = hyper["SigmaInv"]

	phistar = unique(cur["phi"], dims=2)
	nstar = size(phistar)[1]
	if size(phistar) == (2,1)
		phistar = reshape(phistar,:,1)
		nstar = 1
	end

	Sigma1Inv = Sigma0Inv + nstar * SigmaInv 
	Sigma1 = svd2inv(Sigma1Inv) 

	phistarSum = [0.0, 0.0]
	for i in 1:nstar
		phistarSum += log.(phistar[:,i:i])
	end
	mu1 = Sigma1 * (Sigma0Inv * hyper["mu0"] + SigmaInv * phistarSum) # mapslices(sum, log.(phistar), dims=2))

	mu = rand(MvNormal(mu1[:,1], Sigma1), 1)

	return mu
end
