module SparseHop

using OhMyThreads

using LinearAlgebra
using PyPlot
using PottsGauge
using DelimitedFiles
using Random
using Graphs

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
include("sampling.jl")
include("non_sparse.jl")
include("types.jl")
include("utils.jl")
include("dca_score.jl")
include("num_sol.jl")


export Chain, parallel_MCMC, parallel_MCMC_ns, folders, seq_paths_dict, structs_dict, quickread, random_gens, compute_freq, compute_freq!, get_J, get_J!, gibbs_sampling!, dlog, single_kmoh, kmoh!
export dlog!, bisection!, bisection2!, zero_eq!, NumSolVar


end
