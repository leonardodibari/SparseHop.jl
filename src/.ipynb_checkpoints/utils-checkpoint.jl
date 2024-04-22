
compute_freq(Z::Matrix) = compute_weighted_frequencies(Matrix{Int8}(Z), fill(1/size(Z,2), size(Z,2)), 22)

function get_J(K::Array{T,3}, V::Array{T,3}) where {T}
    @tullio J[a, i, b, j] := K[i, j, h] * V[a, b, h] * (j != i)
    return J
end

function get_J!(J::Array{T,4}, K::Array{T,3}, V::Array{T,3}) where {T}
    @tullio J[a, i, b, j] = K[i, j, h] * V[a, b, h] * (j != i)
end

function get_energy(seq::Array{<:Integer,1}, K::Array{T,3}, V::Array{T,3}, h::Array{T,2}) where{T}
    
    J = get_J(K, V)
    @tullio en0[a, i] := J[a, i, seq[j], j] * (i != j)
    en = en0 .+ h
    @tullio res_en := en[seq[i], i]
    
    return -res_en
end


function conn_corr(f1::Array{T, 1}, f2::Array{T, 2}) where {T}
    L = Int(size(f1,1) /21)
    fone = reshape(f1, (21, L))
    ftwo = reshape(f2, (21, L, 21, L))
    @tullio f11[i,a,j,b] := fone[a,i] * fone[b,j]
    return cor(Float32.(f11[:]), Float32.(ftwo[:]))
end

function softmax_notinplace(x::AbstractArray; dims = 1)
    max_ = maximum(x; dims)
    if all(isfinite, max_)
        @fastmath out = exp.(x .- max_)
    else
        @fastmath @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
    end
    return out ./ sum(out; dims)
end

function get_energy(Z::Array{<:Integer,2}, K::Array{T,3}, V::Array{T,3}, h::Array{T,2}) where{T}
    return [get_energy(msa[i,:], K, V, h)  for i in 1:size(Z,2)]
end

function logsumexp(a::AbstractArray{<:Real}; dims=1)
    m = maximum(a; dims=dims)
    return m + log.(sum(exp.(a .- m); dims=dims))
end

function pseudocount1(f1; pc = 0.1, q = 21)
    T = eltype(f1)
    return T.(((1-pc) .* f1 ) .+ (pc / q))
end

function pseudocount2(f2; pc = 0.1, q = 21)
    T = eltype(f2)
    return T.(((1-pc) .* f2 ) .+ (pc / q^2))
end

function Delta_energy(J::Array{T,4}, h::Array{T, 2},
        S::Array{<:Integer,1}, ref::Array{<:Integer, 1}) where {T}
    q, N = size(h)
    E = 0.0
    
    index_v = collect(1:N)
    common = (S .== ref)
    idx = findfirst(isequal(false), common)
    common = index_v[common]
    E -= (h[S[idx],idx] - h[ref[idx],idx])
    @fastmath for j = 1:N
        if j > idx
            @inbounds  E -= (J[S[j],S[idx],j,idx] - J[ref[j],ref[idx],j,idx] )
        end
    end
    
    @fastmath for i in common
        if idx > i
            @inbounds  E -= (J[S[idx],S[i],idx,i] - J[ref[idx],ref[i],idx,i] )
        end
    end
    
    return E
end

function random_gens(num_generators::Int)
    rng_array = []
    for seed in 1:num_generators
        push!(rng_array, Random.MersenneTwister(seed))
    end
    return rng_array
end


function quickread(fastafile; moreinfo=false)  
    Weights, Z, N, M, _ = ReadFasta(fastafile, 0.9, :auto, true, verbose = false);
    moreinfo && return Weights, Z, N, M
    return Matrix{Int8}(Z), Weights
end


function quickread(fastafile, n_seq::Int; moreinfo=false)  
    Weights, Z, N, M, _ = ReadFasta(fastafile, 0.9, :auto, true, n_seq, verbose = false);
    moreinfo && return Weights, Z, N, M
    return Matrix{Int8}(Z), Weights
end

function ReadFasta(filename::AbstractString,max_gap_fraction::Real, theta::Any, remove_dups::Bool;verbose=true)
    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups
        Z, _ = remove_duplicate_sequences(Z,verbose=verbose)
    end
    N, M = size(Z)
    q = round(Int,maximum(Z))
    q > 32 && error("parameter q=$q is too big (max 31 is allowed)")
    W , Meff = compute_weights(Z,q,theta,verbose=verbose)
    println("Meff = $(Meff)")
    rmul!(W, 1.0/Meff)
    Zint=round.(Int,Z)
    return W,Zint,N,M,q
end

function ReadFasta(filename::AbstractString,max_gap_fraction::Real, theta::Any, remove_dups::Bool, n_seq::Int;verbose=true)
    Z = read_fasta_alignment(filename, max_gap_fraction)
    if remove_dups
        Z, _ = remove_duplicate_sequences(Z,verbose=verbose)
    end
    ZZ = Z[:, sample(1:size(Z,2), n_seq, replace=false, ordered=true)]
    N, M = size(Z)
    q = round(Int,maximum(Z))
    q > 32 && error("parameter q=$q is too big (max 31 is allowed)")
    W , Meff = compute_weights(ZZ,q,theta,verbose=verbose)
    println("Meff = $(Meff)")
    rmul!(W, 1.0/Meff)
    Zint=round.(Int,ZZ)
    return W,Zint,N,M,q
end


order = [
    14, 35, 72, 76, 169, 595, 677, 763, 13354,
    90, 105, 131, 200, 412, 593, 1774, 1807, 2953, 7648
]


folders = [
    "../DataAttentionDCA/data/PF00014/",
    "../DataAttentionDCA/data/PF00035/",
    "../DataAttentionDCA/data/PF00072/",
    "../DataAttentionDCA/data/PF00076/",
    "../DataAttentionDCA/data/PF00169/",
    "../DataAttentionDCA/data/PF00595/",
    "../DataAttentionDCA/data/PF00677/",
    "../DataAttentionDCA/data/PF00763/",
    "../DataAttentionDCA/data/PF13354/",
    "../DataAttentionDCA/data/PF00090/",
    "../DataAttentionDCA/data/PF00105/",
    "../DataAttentionDCA/data/PF00131/",
    "../DataAttentionDCA/data/PF00200/",
    "../DataAttentionDCA/data/PF00412/",
    "../DataAttentionDCA/data/PF00593/",
    "../DataAttentionDCA/data/PF01774/",
    "../DataAttentionDCA/data/PF01807/",
    "../DataAttentionDCA/data/PF02953/",
    "../DataAttentionDCA/data/PF07648/"
]

seq_paths = ["../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz",
    "../DataAttentionDCA/data/PF00035/PF00035_full.fasta",
    "../DataAttentionDCA/data/PF00072/PF00072_mgap6.fasta.gz",
    "../DataAttentionDCA/data/PF00076/PF00076_mgap6.fasta.gz",
    "../DataAttentionDCA/data/PF00169/PF00169_full.fasta",
    "../DataAttentionDCA/data/PF00595/PF00595_mgap6.fasta.gz",
    "../DataAttentionDCA/data/PF00677/PF00677_full.fasta",
    "../DataAttentionDCA/data/PF00763/PF00763_full.fasta",
    "../DataAttentionDCA/data/PF13354/PF13354_wo_ref_seqs.fasta.gz", 
    "../DataAttentionDCA/data/PF00090/PF00090.fasta",
"../DataAttentionDCA/data/PF00105/PF00105.fasta",
"../DataAttentionDCA/data/PF00131/PF00131.fasta",
"../DataAttentionDCA/data/PF00200/PF00200.fasta",
"../DataAttentionDCA/data/PF00412/PF00412.fasta",
"../DataAttentionDCA/data/PF00593/PF00593.fasta",
"../DataAttentionDCA/data/PF01774/PF01774.fasta",
"../DataAttentionDCA/data/PF01807/PF01807.fasta",
"../DataAttentionDCA/data/PF02953/PF02953.fasta",
"../DataAttentionDCA/data/PF07648/PF07648.fasta"
]

structs = [
    "../DataAttentionDCA/data/PF00014/PF00014_struct.dat",
    "../DataAttentionDCA/data/PF00035/Atomic_distances_PF00035.dat",
    "../DataAttentionDCA/data/PF00072/PF00072_struct.dat",
    "../DataAttentionDCA/data/PF00076/PF00076_struct.dat",
    "../DataAttentionDCA/data/PF00169/Atomic_distances_PF00169.dat",
    "../DataAttentionDCA/data/PF00595/PF00595_struct.dat",
    "../DataAttentionDCA/data/PF00677/Atomic_distances_PF00677.dat",
    "../DataAttentionDCA/data/PF00763/Atomic_distances_PF00763.dat",
    "../DataAttentionDCA/data/PF13354/PF13354_struct.dat",
    "../DataAttentionDCA/data/PF00090/PF00090_struct_struct.dat",
    "../DataAttentionDCA/data/PF00105/PF00105_struct_struct.dat",
    "../DataAttentionDCA/data/PF00131/PF00131_struct_struct.dat",
    "../DataAttentionDCA/data/PF00200/PF00200_struct_struct.dat",
    "../DataAttentionDCA/data/PF00412/PF00412_struct_struct.dat",
    "../DataAttentionDCA/data/PF00593/PF00593_struct_struct.dat",
    "../DataAttentionDCA/data/PF01774/PF01774_struct_struct.dat",
    "../DataAttentionDCA/data/PF01807/PF01807_struct_struct.dat",
    "../DataAttentionDCA/data/PF02953/PF02953_struct_struct.dat",
    "../DataAttentionDCA/data/PF07648/PF07648_struct_struct.dat"
]
# Create dictionaries
folders_dict = Dict(zip(order, folders))
seq_paths_dict = Dict(zip(order, seq_paths))
structs_dict = Dict(zip(order, structs))
