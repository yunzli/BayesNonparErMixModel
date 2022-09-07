module BayesErMixRegModel

using JLD2, ProgressBars, TOML
using Random, StatsBase, LinearAlgebra, Distributions
using RCall

Random.seed!(20220901)

include("./MCMC.jl")
include("./estimation.jl")
include("./update_M.jl")
include("./update_theta.jl")
include("./update_mu.jl")
include("./update_phi.jl")
include("../nocovariate/update_alpha.jl")
include("../utils.jl")

end
