module BayesErMixModel 

using JLD2, ProgressBars, TOML
using Random, StatsBase, LinearAlgebra, Distributions
using RCall

Random.seed!(20220901)

include("./update_theta.jl")
include("./update_theta_M.jl")
include("./update_M.jl")
include("./update_zeta.jl")
include("./update_phi.jl")
include("./update_alpha.jl") 
include("./MCMC.jl")
include("./estimation.jl")
include("../utils.jl")

end 
