function _estimation(M, theta, mu, sigma2, phi, alpha, survival, nu, grids)

	ngrids = length(grids)
	nkeep = length(M) 
	densEval = zeros(nkeep, ngrids)
	survEval = zeros(nkeep, ngrids)
	hazaEval = zeros(nkeep, ngrids)

	n = length(survival)
	eff_w_sum = zeros(nkeep)
	eff_w_len = zeros(nkeep)

	for j in ProgressBar(1:nkeep)

		G0, D = get_G0_and_D(M[j], LogNormal(mu[j], sqrt(Sigma)), theta[j])

		intervals = [theta[j]:theta[j]:(M[j]-1)*theta[j];]

		deltas = zeros(Int64, M[j])
		for i in 1:n
			idx = findInterval(phi[j,i], intervals)
			deltas[idx] += 1
		end

		params = zeros(M[j])
		Z = zeros(M[j])

		for m in 1:M[j]
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
		#  print(sum(w_tmp[findall(w_tmp .> 0.01)]), "\n")

		for g in 1:ngrids
			for m in 1:M[j]
				densEval[j, g] += w[m] * pdf(Gamma(m, theta[j]), grids[g])[1]
				survEval[j, g] += w[m] * ccdf(Gamma(m,theta[j]), grids[g])[1]
			end
			hazaEval[j,g] = densEval[j,g] / survEval[j,g]
		end
	end

	return Dict("densEval" => densEval,
				"survEval" => survEval,
				"hazaEval" => hazaEval,
				"eff_w_len" => eff_w_len, 
				"eff_w_sum" => eff_w_sum)
end

function estimation(config_file, is_estimate=true)

	config = TOML.parsefile(config_file)

	simdataRaw = load( config["data_path"] * config["datafile"] )
	simdata = preprocessor(simdataRaw)
	survivals = simdata["survival"]
	fitdata = load( config["save_path"] * "fit_" * config["datafile"] )

	pos = fitdata["pos"]
	hyper = config["hyper"]
	hyper["Sigma"] = mapreduce(permutedims, vcat, hyper["Sigma"]) # Vec[Vec] to Matrix 

	n = length(simdata["survival"])
	nsam = length(pos["alpha"])
	nburn = div(nsam, 4)
	nthin = div(nsam-nburn, 2000)

	keep_index = [nburn+1:nthin:nsam;]
   	nkeep = length(keep_index)

	print("burn: ", nburn, " thin: ", nthin, " keep: ", nkeep, "\n")
	
	theta_save = pos["theta"][keep_index,:]
	M_save = pos["M"][keep_index,:]
	alpha_save = pos["alpha"][keep_index] 
	phi_save = pos["phi"][keep_index,:,:]
	mu_save = pos["mu"][keep_index,:]

	survivalC = simdata["survivalC"]
	survivalT = simdata["survivalT"]
	indexC = simdata["indexC"]
	indexT = simdata["indexT"]
	nuC = simdata["nuC"]
	nuT = simdata["nuT"]

	fig_path = config["fig_path"]
	if !isdir(fig_path)
		mkdir(fig_path)
	end
	@rput theta_save M_save alpha_save mu_save phi_save fig_path 
	R"""
	pdf(paste0(fig_path, "trace.pdf"))
	plot(theta_save[,1], type='l', main="thetaC")
	plot(theta_save[,2], type='l', main="thetaT")
	plot(alpha_save, type='l', main="alpha")
	plot(mu_save[,1], type='l', main="mu1")
	plot(mu_save[,2], type='l', main="mu2")
	plot(M_save[,1], type='l', main="M1")
	plot(M_save[,2], type='l', main="M2")
	dev.off()
	pdf(paste0(fig_path, "hist.pdf"))
	hist(theta_save[,1], main="thetaC")
	hist(theta_save[,2], main="thetaT")
	hist(alpha_save, main="alpha")
	hist(mu_save[,1], main="mu1")
	hist(mu_save[,2], main="mu2")
	hist(M_save[,1], main="M1")
	hist(M_save[,2], main="M2")
    hist(phi_save[1,,], main="phi") 
	dev.off()
	"""

	gridsC = range(0.001, maximum(survivalC), length=1000)
	gridsT = range(0.001, maximum(survivalT), length=1000)

	savefile = config["save_path"] * "inference_" * config["datafile"]
	if is_estimate 
		estimationsC = _estimation(M_save[:,1], theta_save[:,1], mu_save[:,1], hyper["Sigma"][1,1], phi_save[:,1,indexC], alpha_save, survivalC, nuC, gridsC)
		estimationsT = _estimation(M_save[:,2], theta_save[:,2], mu_save[:,2], hyper["Sigma"][2,2], phi_save[:,2,indexT], alpha_save, survivalT, nuT, gridsT)
		result = Dict("control" => estimationsC,
					  "treatment" => estimationsT) 
		save(savefile, result)
	else
		result = load(savefile)
		estimationsC = result["control"]
		estimationsT = result["treatment"] 
	end

	eff_w_lenC = estimationsC["eff_w_len"]
	eff_w_lenT = estimationsT["eff_w_len"]
	eff_w_sumC = estimationsC["eff_w_sum"]
	eff_w_sumT = estimationsT["eff_w_sum"]
	print("average: ", mean(eff_w_lenC), " ", mean(eff_w_lenT), " ", mean(eff_w_sumC), " ", mean(eff_w_sumT), "\n")

	@rput eff_w_lenC eff_w_lenT eff_w_sumC eff_w_sumT 
	R"""
	pdf(paste0(fig_path, "eff_w.pdf"))
	hist(eff_w_lenC)
	hist(eff_w_lenT)
	hist(eff_w_sumC)
	hist(eff_w_sumT)
	dev.off()
	"""

	densEvalC = estimationsC["densEval"]
	densEvalT = estimationsT["densEval"]
	survEvalC = estimationsC["survEval"]
	survEvalT = estimationsT["survEval"]
	hazaEvalC = estimationsC["hazaEval"]
	hazaEvalT = estimationsT["hazaEval"]

	@rput densEvalC survEvalC hazaEvalC gridsC survivalC nuC 
	@rput densEvalT survEvalT hazaEvalT gridsT survivalT nuT 
	R"""
	library(ggplot2)

	survivalC = data.frame(x=survivalC, status=nuC)
	survivalT = data.frame(x=survivalT, status=nuT)

	#  conf_interval = c(0.005, 0.995)
	conf_interval = c(0.025, 0.975)
	densMeanC = apply(densEvalC, 2, mean)
	densQuanC = apply(densEvalC, 2, quantile, prob=conf_interval)
	survMeanC = apply(survEvalC, 2, mean)
	survQuanC = apply(survEvalC, 2, quantile, prob=conf_interval)
	hazaMeanC = apply(hazaEvalC, 2, mean)
	hazaQuanC = apply(hazaEvalC, 2, quantile, prob=conf_interval)

	densMeanT = apply(densEvalT, 2, mean)
	densQuanT = apply(densEvalT, 2, quantile, prob=conf_interval)
	survMeanT = apply(survEvalT, 2, mean)
	survQuanT = apply(survEvalT, 2, quantile, prob=conf_interval)
	hazaMeanT = apply(hazaEvalT, 2, mean)
	hazaQuanT = apply(hazaEvalT, 2, quantile, prob=conf_interval)

	densC = data.frame(x=gridsC, e=densMeanC, l=densQuanC[1,], r=densQuanC[2,])
	survC = data.frame(x=gridsC, e=survMeanC, l=survQuanC[1,], r=survQuanC[2,])
	hazaC = data.frame(x=gridsC, e=hazaMeanC, l=hazaQuanC[1,], r=hazaQuanC[2,])

	densT = data.frame(x=gridsT, e=densMeanT, l=densQuanT[1,], r=densQuanT[2,])
	survT = data.frame(x=gridsT, e=survMeanT, l=survQuanT[1,], r=survQuanT[2,])
	hazaT = data.frame(x=gridsT, e=hazaMeanT, l=hazaQuanT[1,], r=hazaQuanT[2,])

	p0 = ggplot(densC) + geom_ribbon(aes(x=x, ymin=l, ymax=r), fill="red", alpha=0.4) + geom_line(aes(x=x,y=e), color="red", linetype="dashed") + labs(x="t", y="Density")
	if( sum(nuC == 0) == 0 ){
		p0 = p0 + geom_rug(data=survivalC, aes(x=x), color="black", alpha=0.5, show.legend = FALSE, inherit.aes=FALSE) 
	}else{
		p0 = p0 + geom_rug(data=subset(survivalC, status == 1), aes(x=x), color="black", alpha=0.5, show.legend = FALSE, inherit.aes=FALSE) 
		p0 = p0 + geom_rug(data=subset(survivalC, status == 0), aes(x=x), color="red", alpha=0.5, show.legend = FALSE, inherit.aes=FALSE) 
	}
	p1 = ggplot(survC) + geom_ribbon(aes(x=x, ymin=l, ymax=r), fill="red", alpha=0.4) + geom_line(aes(x=x,y=e), color="red", linetype="dashed") + labs(x="t", y="Survival")
	p2 = ggplot(hazaC) + geom_ribbon(aes(x=x, ymin=l, ymax=r), fill="red", alpha=0.4) + geom_line(aes(x=x,y=e), color="red", linetype="dashed") + labs(x="t", y="Hazard")

	p3 = ggplot(densT) + geom_ribbon(aes(x=x, ymin=l, ymax=r), fill="blue", alpha=0.4) + geom_line(aes(x=x,y=e), color="blue", linetype="dashed") + labs(x="t", y="Density")
	if( sum(nuT == 0) == 0 ){
		p0 = p0 + geom_rug(data=survivalT, aes(x=x), color="black", alpha=0.5, show.legend = FALSE, inherit.aes=FALSE) 
	}else{
		p0 = p0 + geom_rug(data=subset(survivalT, status == 1), aes(x=x), color="black", alpha=0.5, show.legend = FALSE, inherit.aes=FALSE) 
		p0 = p0 + geom_rug(data=subset(survivalT, status == 0), aes(x=x), color="red", alpha=0.5, show.legend = FALSE, inherit.aes=FALSE) 
	}
	p4 = ggplot(survT) + geom_ribbon(aes(x=x, ymin=l, ymax=r), fill="blue", alpha=0.4) + geom_line(aes(x=x,y=e), color="blue", linetype="dashed") + labs(x="t", y="Survival")
	p5 = ggplot(hazaT) + geom_ribbon(aes(x=x, ymin=l, ymax=r), fill="blue", alpha=0.4) + geom_line(aes(x=x,y=e), color="blue", linetype="dashed") + labs(x="t", y="Hazard")
	
	p6 = ggplot() + labs(x="t", y="Survival")
	p6 = p6 + geom_ribbon(data=survC, aes(x=x, ymin=l, ymax=r), fill="red",  alpha=0.4) + geom_line(data=survC, aes(x=x,y=e), color="red",  linetype="dashed") 
	p6 = p6 + geom_ribbon(data=survT, aes(x=x, ymin=l, ymax=r), fill="blue", alpha=0.4) + geom_line(data=survT, aes(x=x,y=e), color="blue", linetype="dashed")
	
	p7 = ggplot() + labs(x="t", y="Hazard")
	p7 = p7 + geom_ribbon(data=hazaC, aes(x=x, ymin=l, ymax=r), fill="red",  alpha=0.4) + geom_line(data=hazaC, aes(x=x,y=e), color="red",  linetype="dashed") 
	p7 = p7 + geom_ribbon(data=hazaT, aes(x=x, ymin=l, ymax=r), fill="blue", alpha=0.4) + geom_line(data=hazaT, aes(x=x,y=e), color="blue", linetype="dashed")
	"""

	if haskey(simdata, "modelC")
		densTrueC = pdf.(simdata["modelC"], gridsC)
		survTrueC = ccdf.(simdata["modelC"], gridsC)
		hazaTrueC = densTrueC ./ survTrueC
		densTrueT = pdf.(simdata["modelT"], gridsT)
		survTrueT = ccdf.(simdata["modelT"], gridsT)
		hazaTrueT = densTrueT ./ survTrueT
		@rput densTrueC survTrueC hazaTrueC
		@rput densTrueT survTrueT hazaTrueT
		R"""
		densC["t"]=densTrueC 
		survC["t"]=survTrueC 
		hazaC["t"]=hazaTrueC 
		densT["t"]=densTrueT 
		survT["t"]=survTrueT 
		hazaT["t"]=hazaTrueT 
		p0 = p0 + geom_line(data=densC, aes(x=x,y=t), color="red")
		p1 = p1 + geom_line(data=survC, aes(x=x,y=t), color="red")
		p2 = p2 + geom_line(data=hazaC, aes(x=x,y=t), color="red")
		p3 = p3 + geom_line(data=densT, aes(x=x,y=t), color="blue")
		p4 = p4 + geom_line(data=survT, aes(x=x,y=t), color="blue")
		p5 = p5 + geom_line(data=hazaT, aes(x=x,y=t), color="blue")

		p6 = p6 + geom_line(data=survC, aes(x=x,y=t), color="red")
		p6 = p6 + geom_line(data=survT, aes(x=x,y=t), color="blue")

		p7 = p7 + geom_line(data=hazaC, aes(x=x,y=t), color="red")
		p7 = p7 + geom_line(data=hazaT, aes(x=x,y=t), color="blue")
		"""
	end       

	R"""
	ggsave(paste0(fig_path,"densC.png"), p0)
	ggsave(paste0(fig_path,"survC.png"), p1)
	ggsave(paste0(fig_path,"hazaC.png"), p2)
	ggsave(paste0(fig_path,"densT.png"), p3)
	ggsave(paste0(fig_path,"survT.png"), p4)
	ggsave(paste0(fig_path,"hazaT.png"), p5)
	ggsave(paste0(fig_path,"surv.png"), p6)
	ggsave(paste0(fig_path,"haza.png"), p7)
	"""

end
