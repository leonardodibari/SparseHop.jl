module SparseHop

using OhMyThreads
using Base.Threads
using LinearAlgebra
using PyPlot
using PottsGauge
using DelimitedFiles
using Random
using Graphs
using Optim
using ComponentArrays
using Zygote, JLD2
using LineSearches
using NLopt

import LogExpFunctions: logsumexp
import Tullio: @tullio
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
import Printf:@printf
import Statistics: mean, cor
import StatsBase: sample, weights
import Flux: softmax!, Adam
import Flux.Optimise: update! 
import Flux.Optimiser

include("sparse.jl")
include("new_sparse.jl")
include("sampling.jl")
include("non_sparse.jl")
include("types.jl")
include("utils.jl")
include("dca_score.jl")
include("num_sol.jl")
include("minimization.jl")
include("nlopt_minim.jl")

include("learn_V.jl")


export Chain, Data, ModelData, NumSolVar 
export folders, seq_paths_dict, structs_dict 
export get_dlog!, activate_edges!, zero_eq, dlog
export run_gibbs_sampling!, update_ModelData!, grad_update!
export parallel_MCMC, runSparseHop, new_runSparseHop, new_runSparseHop_onV


end
