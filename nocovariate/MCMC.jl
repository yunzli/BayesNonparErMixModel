function MCMC(config_file)

	config = TOML.parsefile(config_file)

	dat = load(config["data_path"] * config["datafile"])
	hyper = config["hyper"]

	n = length(dat["survival"])
	cur = Dict("zeta"=> rand(InverseGamma(hyper["a_zeta"], hyper["b_zeta"]),1)[1],
			   "M" => sample([ceil(Int32,hyper["M1"]/(hyper["a_theta"]*hyper["b_theta"])):1:ceil(Int32, hyper["M2"]/(hyper["a_theta"]*hyper["b_theta"]));]),
			   "alpha" => 2,
			   "phi" => rand(Exponential(5),n),
			   "theta" => rand(Gamma(hyper["a_theta"], hyper["b_theta"]),1)[1]
			   )

	batch_size = 50 
	nbatch = config["nbatch"] 
	nsam = nbatch * batch_size 

	pos = Dict("zeta" => Vector{Float64}(undef, nsam),
			   "M" => Vector{Int32}(undef, nsam),
			   "alpha" => Vector{Float64}(undef, nsam),
			   "phi" => Matrix{Float64}(undef, nsam, n),
			   "theta" => Vector{Float64}(undef, nsam)
			   )

	l_rate = 0.0
	for n_b in ProgressBar(1:nbatch)
	#  for n_b in (1:nbatch)

		count = 0 
		for i_t in 1:batch_size 
			i = (n_b-1)*batch_size + i_t 

			cur["theta"], cur["M"] = update_theta_M(cur, dat, hyper, l_rate)
			
			tmp = update_theta(cur, dat, hyper, l_rate)
			count += tmp["acc"]
			pos["theta"][i] = cur["theta"] = tmp["theta"] 

			pos["M"][i] = cur["M"] = update_M(cur, dat, hyper) 
			
			pos["zeta"][i] = cur["zeta"] = update_zeta(cur, dat, hyper)
			pos["phi"][i,:] = cur["phi"] = update_phis(cur, dat, hyper)
			pos["alpha"][i] = cur["alpha"] = update_alpha(cur, dat, hyper) 

		end

		acc = count / batch_size
		delta_n = minimum([0.01, n_b^(-0.5)])

		if acc <= 0.44
			l_rate -= delta_n
		else
			l_rate += delta_n
		end
		#  print("\nl_rate: ", l_rate, " acc", acc, "\n")
	end 

	result = Dict("pos" => pos,
				  "hyper" => hyper)

	savefile = config["save_path"] * "fit_" * config["datafile"]
	save(savefile, result) 

end

