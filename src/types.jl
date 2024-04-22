struct Chain{T}
    seq::Array{<:Integer,1}
    L::Int
    log_prob::Array{T,1}
    prob::Array{T,1}
    generator
end

function Chain(seq::Array{<:Integer,1}, generator; T::DataType=Float32)
    L = size(seq,1)
    log_prob = T.(zeros(21))
    prob = T.(zeros(21))
    Chain{T}(seq, L, log_prob, prob, generator)
end


struct Data{T, T2}
    msa::Array{T2,2}
    f1::Array{T,1}
    f2::Array{T,2}
    f1rs::Array{T,2}
    f2rs::Array{T,4}
    V::Array{T,3}
    mheads::Array{T,3}
end

function Data(filepath, V; T::DataType=Float32, T2::DataType = Int8)
    msa = T2.(quickread(filepath)[1])
    L = size(msa,1)
    f1, f2, M = compute_weighted_frequencies(msa, 22, 0.2)
    f1 = T.(f1)
    f2 = T.(f2)
    f1rs = reshape(f1, (21, L))
    f2rs = reshape(f2, (21, L, 21, L))
    mheads = T.(zeros(L,L,size(V,3)))
    moh!(mheads, pseudocount2(f2rs, pc = 0.1), V)
    Data{T, T2}(msa, f1, f2, f1rs, f2rs, V, mheads)
end

struct ModelSample{T}
    msa::Array{<:Integer,2}
    f1::Array{T,1}
    f2::Array{T,2}
    f1rs::Array{T,2}
    f2rs::Array{T,4}
    V::Array{T,3}
    mheads::Array{T,3}
    sqheads::Array{T,3}
end

function ModelSample(chains, V, L::Int, N_chains::Int; T::DataType=Float32)
    
    msa = Int8.(zeros(L, N_chains))
    for n in 1:N_chains
        msa[:,n] = Int8.(chains[n].seq)
    end
    
    f1, f2 = compute_freq(Int8.(msa))
    f1 = T.(f1)
    f2 = T.(f2)
    f1rs = reshape(f1, (21, L))
    f2rs = reshape(f2, (21, L, 21, L))
    mheads = T.(zeros(L,L,size(V,3)))
    moh!(mheads, f2rs, V)
    sqheads = T.(zeros(L,L,size(V,3)))
    soh!(sqheads, f2rs, V)
    ModelSample{T}(msa, f1, f2, f1rs, f2rs, V, mheads, sqheads)
end

function update_sample!(msa::Array{<:Integer,2},
    f1::Array{T,1},
    f2::Array{T,2},
    f1rs::Array{T,2},
    f2rs::Array{T,4},
    V::Array{T,3},
    mheads::Array{T,3},
    sqheads::Array{T,3},
    chains, 
    L::Int, 
    N_chains::Int) where {T}
    
    for n in 1:N_chains
        msa[:,n] = Int8.(chains[n].seq)
    end
    
    TT = eltype(f1)
    f1, f2 = compute_freq(Int8.(msa))
    f1 = TT.(f1)
    f2 = TT.(f2)
    f1rs = reshape(f1, (21, L))
    f2rs = reshape(f2, (21, L, 21, L))
    moh!(mheads, pseudocount2(f2rs, pc = 0.1), V)
    soh!(sqheads, pseudocount2(f2rs, pc = 0.1), V)
end
    

    
    
    

