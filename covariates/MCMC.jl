function preprocessor(dat)
	"""
	This function extracts group information and returns a dictionary 
	"""
	indC = findall(x->x==0, dat["group"])
	indT = findall(x->x==1, dat["group"])

	nuC = dat["nu"][indC]
	nuT = dat["nu"][indT]

	survivalC = dat["survival"][indC]
	survivalT = dat["survival"][indT]

	if haskey(dat, "modelC")
		result = Dict(
					  "indexC" => indC, 
					  "indexT" => indT, 
					  "survivalC" => survivalC, 
					  "survivalT" => survivalT, 
					  "survival" => dat["survival"], 
					  "nuC" => nuC, 
					  "nuT" => nuT,
					  "nu" => dat["nu"],
					  "modelC" => dat["modelC"],
					  "modelT" => dat["modelT"],
					  "group" => dat["group"]
					 )
	else
		result = Dict(
					  "indexC" => indC, 
					  "indexT" => indT, 
					  "survivalC" => survivalC, 
					  "survivalT" => survivalT, 
					  "survival" => dat["survival"], 
					  "nuC" => nuC, 
					  "nuT" => nuT,
					  "nu" => dat["nu"],
					  "group" => dat["group"]
					 )
	end
    
	return result 
end 

function MCMC(config_file)

	config = TOML.parsefile(config_file)

	datRaw = load(config["data_path"] * config["datafile"])
	dat = preprocessor(datRaw) 
	
	hyper = config["hyper"]

	nC = length(dat["survivalC"])
	nT = length(dat["survivalT"])
	n = nC + nT 

	hyper["Sigma0"] = mapreduce(permutedims, vcat, hyper["Sigma0"]) # Vec[Vec] to Matrix
	hyper["Sigma0Inv"] = svd2inv(hyper["Sigma0"])
	hyper["Sigma"] = mapreduce(permutedims, vcat, hyper["Sigma"]) # Vec[Vec] to Matrix 
	hyper["SigmaInv"] = svd2inv(hyper["Sigma"])

	cur = Dict("mu" => hyper["mu0"],
			   "M" => [100, 100],
			   "alpha" => 2.0,
			   "phi" => rand(MvLogNormal(hyper["mu0"], hyper["Sigma"]),n),
			   "theta" => [hyper["a_C"]*hyper["b_C"],
						   hyper["a_T"]*hyper["b_T"]]
			   )

	batch_size = 50 
	nbatch = config["nbatch"] 
	nsam = nbatch * batch_size 

	pos = Dict("mu" => Matrix{Float64}(undef, nsam, 2),
			   "M" => Matrix{Int64}(undef, nsam, 2),
			   "theta" => Matrix{Float64}(undef, nsam, 2),
			   "alpha" => Vector{Float64}(undef, nsam),
			   "phi" => Array{Float64}(undef, nsam, 2, n),
			   )

	Sig = 0.01*[1.0 0.0; 0.0 1.0]
	for n_b in ProgressBar(1:nbatch)

		for i_t in 1:batch_size 

			i = (n_b-1)*batch_size + i_t 

			pos["theta"][i,:] = cur["theta"] = update_theta(cur, dat, hyper, Sig) 
			pos["M"][i,:] = cur["M"] = update_M(cur, dat, hyper) 
			pos["phi"][i,:,:] = cur["phi"] = update_phis(cur, dat, hyper)
			pos["mu"][i,:] = cur["mu"] = update_mu(cur, dat, hyper)
			pos["alpha"][i] = cur["alpha"] = update_alpha(cur, dat, hyper) 

		end
		#  print("theta :", cur["theta"], "\n\n")
		Sig = cov(log.(pos["theta"][1:(n_b*batch_size),:])) + 1.0e-10*Diagonal(ones(2)) # adaptive MCMC for theta

	end 

	result = Dict("pos" => pos)

	savefile = config["save_path"] * "fit_" * config["datafile"]
	save(savefile, result) 

end

