module SparseHop

using OhMyThreads

using LinearAlgebra
using PyPlot
using PottsGauge
using DelimitedFiles
using Random
using Graphs

import Tullio: @tullio
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
import Printf:@printf
import Statistics: mean, cor
import StatsBase: sample, weights

include("sparse.jl")
include("types.jl")
include("utils.jl")
include("dca_score.jl")


export Chain, parallel_MCMC, folders, seq_paths_dict, structs_dict, quickread, random_gens, compute_freq


end
