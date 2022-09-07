function _estimation(M, theta, zeta, phi, alpha, survival, nu, grids)

	ngrids = length(grids) 
	nkeep  = length(M) 
	densEval = zeros(nkeep, ngrids)
	survEval = zeros(nkeep, ngrids)
	hazaEval = zeros(nkeep, ngrids) 

	n = length(survival)
	eff_w_sum = zeros(nkeep)
	eff_w_len = zeros(nkeep)

	for j in ProgressBar(1:nkeep)

		G0, D = get_G0_and_D(M[j], Exponential(zeta[j]), theta[j])

		intervals = [theta[j]:theta[j]:(M[j]-1)*theta[j];]

		deltas = zeros(Int64, M[j])
		for i in 1:n
			idx = findInterval(phi[j,i], intervals)
			deltas[idx] += 1
		end

		params = zeros(M[j]) # alpha[j] * D .+ deltas 
		Z = zeros(M[j])

		for m = 1:M[j]
			params[m] = alpha[j] * D[m] + deltas[m]
			if params[m] != 0
				Z[m] = rand(Gamma(params[m], 1),1)[1]
			end
		end 
		Z_sum = sum(Z) 
		w = Z ./ Z_sum 
        w_tmp = w[findall(w .> 0.01)]
		eff_w_sum[j] = sum(w_tmp)
		eff_w_len[j] = length(w_tmp) 

		for g in 1:ngrids 
			for m in 1:M[j]
				densEval[j,g] += w[m] * pdf( Gamma(m, theta[j]), grids[g] )[1]
				survEval[j,g] += w[m] * ccdf( Gamma(m, theta[j]), grids[g] )[1]
			end
			hazaEval[j,g] = densEval[j,g] / survEval[j,g]
		end
	end
	return Dict("densEval" => densEval, 
				"survEval" => survEval,
				"hazaEval" => hazaEval,
				"eff_w_sum" => eff_w_sum,
				"eff_w_len" => eff_w_len 
				)
end

function estimation(config_file, is_estimate=true)

	config = TOML.parsefile(config_file)

	simdata = load( config["data_path"] * config["datafile"] )
	survivals = simdata["survival"]
	nu = simdata["nu"]
	fitdata = load( config["save_path"] * "fit_" * config["datafile"] )

	pos = fitdata["pos"]
	hyper = fitdata["hyper"]

	n = length(survivals)
	nsam = length(pos["alpha"])
	nburn = div(nsam, 4)
	nthin = div(nsam-nburn, 2000)

	keep_index = [nburn+1:nthin:nsam;]
   	nkeep = length(keep_index)

	print("burn: ", nburn, " thin: ", nthin, " keep: ", nkeep, "\n")

	theta_save = pos["theta"][keep_index]
	zeta_save = pos["zeta"][keep_index]
	M_save = pos["M"][keep_index]
	alpha_save = pos["alpha"][keep_index]
	phi_save = pos["phi"][keep_index,:]

	phi_sample = zeros(nkeep)
	for i in 1:nkeep
		phi_sample[i] = sample(phi_save[i,:])
	end
	@rput phi_sample

	fig_path = config["fig_path"]
	if !isdir(fig_path) 
		mkdir(fig_path) 
	end 
	@rput theta_save zeta_save M_save alpha_save survivals fig_path
	R"""
	pdf(paste0(fig_path, "trace.pdf"))
	plot(theta_save, type='l', main="theta")
	plot(zeta_save, type='l', main="b")
	plot(alpha_save, type='l', main="alpha")
	plot(M_save, type='l', main="M")
	hist(survivals)
	dev.off()
	pdf(paste0(fig_path, "hist.pdf"))
	hist(theta_save)
	hist(zeta_save)
	hist(alpha_save)
	hist(M_save)
	hist(phi_sample) 
	dev.off()
	"""
	
	if haskey(simdata, "x_max")
		grids = range(1e-2, simdata["x_max"], length=1000)
	else
		grids = range(1e-2, Base.maximum(simdata["survival"]), length=1000)
	end

	savefile = config["save_path"] * "inference_" * config["datafile"]
	if is_estimate
		estimations = _estimation(M_save, theta_save, zeta_save, phi_save, alpha_save, survivals, nu, grids) 
	    save(savefile, estimations)
	else
		estimations = load(savefile) 
	end
	densEval  = estimations["densEval"]
	survEval  = estimations["survEval"]
	hazaEval  = estimations["hazaEval"]
	eff_w_len = estimations["eff_w_len"]
	eff_w_sum = estimations["eff_w_sum"]

	print("average: ", mean(eff_w_len), " ", mean(eff_w_sum), "\n")
	@rput densEval survEval hazaEval grids survivals nu
	R"""
	library(ggplot2)

	survivals = data.frame(x=survivals, status=nu)
	conf_interval = c(0.025, 0.975)
	densMean = apply(densEval, 2, mean)
	densQuan = apply(densEval, 2, quantile, prob=conf_interval)
	survMean = apply(survEval, 2, mean)
	survQuan = apply(survEval, 2, quantile, prob=conf_interval)
	hazaMean = apply(hazaEval, 2, mean)
	hazaQuan = apply(hazaEval, 2, quantile, prob=conf_interval)

	dens = data.frame(x=grids, e=densMean, l=densQuan[1,], r=densQuan[2,])
	surv = data.frame(x=grids, e=survMean, l=survQuan[1,], r=survQuan[2,])
	haza = data.frame(x=grids, e=hazaMean, l=hazaQuan[1,], r=hazaQuan[2,])

	p0 = ggplot(dens) + geom_ribbon(aes(x=x, ymin=l, ymax=r), fill="red", alpha=0.4) + geom_line(aes(x=x,y=e), color="red", linetype="dashed") + labs(y="Density", x="t")
	if( sum(nu == 0) == 0 ){
		p0 = p0 + geom_rug(data=survivals, aes(x=x), color="black", alpha=0.5, show.legend = FALSE, inherit.aes=FALSE) 
	}else{
		p0 = p0 + geom_rug(data=subset(survivals, status == 1), aes(x=x), color="black", alpha=0.5, show.legend = FALSE, inherit.aes=FALSE) 
		p0 = p0 + geom_rug(data=subset(survivals, status == 0), aes(x=x), color="red", alpha=0.5, show.legend = FALSE, inherit.aes=FALSE) 
	}

	p1 = ggplot(surv) + geom_ribbon(aes(x=x, ymin=l, ymax=r), fill="red", alpha=0.4) + geom_line(aes(x=x,y=e), color="red", linetype="dashed") + labs(y="Survival", x="t")

	p2 = ggplot(haza) + geom_ribbon(aes(x=x, ymin=l, ymax=r), fill="red", alpha=0.4) + geom_line(aes(x=x,y=e), color="red", linetype="dashed") + labs(y="Hazard", x="t")
	"""
	
	if haskey(simdata, "model")
		densTrue = pdf.(simdata["model"], grids)
		survTrue = ccdf.(simdata["model"], grids)
		hazaTrue = densTrue ./ survTrue 
		@rput densTrue survTrue hazaTrue
		R"""
        dens["t"] = densTrue 
		surv["t"] = survTrue 
        haza["t"] = hazaTrue 
		p0 = p0 + geom_line(data=dens, aes(x=x,y=t), color="red")
		p1 = p1 + geom_line(data=surv, aes(x=x,y=t), color="red")
		p2 = p2 + geom_line(data=haza, aes(x=x,y=t), color="red")
		"""
	end
	
	R"""
	ggsave(paste0(fig_path,"dens.png"), p0)
	ggsave(paste0(fig_path,"surv.png"), p1)
	ggsave(paste0(fig_path,"haza.png"), p2)

	#  hazardTest = densMean / survMean
    #  hazaTest = data.frame(x=grids, y=hazardTest)
	#  ptest = ggplot(hazaTest) + geom_line(aes(x=x,y=y)) + geom_line(data=haza, aes(x=x,y=t), color="red")
	#  ggsave(paste0(fig_path,"test.png"), ptest)
	"""
end
