struct Chain{T}
    seq::Array{<:Integer,1}
    aminos::Array{<:Integer,1}
    sites::Array{<:Integer,1}
    L::Int
    log_prob::Array{T,1}
    max_::T
    generator
end

function Chain(seq::Array{<:Integer,1}, q::Int, generator; T::DataType=Float32)
    L = size(seq,1)
    log_prob = T.(zeros(q))
    aminos = [i for i in 1:q]
    sites = [i for i in 1:L]
    max_ = T(0)
    Chain{T}(seq, aminos, sites, L, log_prob, max_, generator)
end


struct Data{T, T2}
    msa::Array{T2,2}
    f1::Array{T,1}
    f2::Array{T,2}
    f1rs::Array{T,2}
    f2rs::Array{T,4}
    f2rspc::Array{T,4}
    V::Array{T,3}
    mheads::Array{T,3}
    mheadspc::Array{T,3}
end

function Data(filepath, V, pc::AbstractFloat, q::Int, H::Int; T::DataType=Float32, T2::DataType = Int8)
    msa = T2.(quickread(filepath)[1])
    L = size(msa,1)
    f1, f2, M = compute_weighted_frequencies(msa, q+1, 0.2)
    f1 = T.(f1)
    f2 = T.(f2)
    f1rs = reshape(f1, (q, L))
    f2rs = reshape(f2, (q, L, q, L))
    f2rspc = pseudocount2(f2rs, T, pc, q)
    mheads = T.(zeros(L,L,size(V,3)))
    mheadspc = T.(zeros(L,L,size(V,3)))
    moh!(mheads, f2rs, V, L, H, q)
    moh!(mheadspc, f2rspc, V, L, H, q)
    Data{T, T2}(msa, f1, f2, f1rs, f2rs, f2rspc, V, mheads, mheadspc)
end


struct NumSolVar{T}
    k::Array{T,1}
    dest::Array{T,1}
    vars::Array{T,1}
end

function NumSolVar(T::DataType=Float32)
    k = T.(zeros(3))
    dest = T.(zeros(3))
    vars = T.(zeros(2))
    NumSolVar{T}(k, dest, vars)
end


function update_sample!(msa::Array{<:Integer,2},
    f1::Array{T,1},
    f2::Array{T,2},
    f1rs::Array{T,2},
    f2rs::Array{T,4},
    f2rspc::Array{T,4},
    V::Array{T,3},
    mheads::Array{T,3},
    mheadspc::Array{T,3},
    L::Int,
    pc::T,
    q::Int,
    H::Int,
    TT) where {T}
    
    f1, f2 = compute_freq(Int8.(msa))
    f1 .= TT.(f1)
    f2 .= TT.(f2)
    f1rs .= reshape(f1, (q, L))
    f2rs .= reshape(f2, (q, L, q, L))
    pseudocount2!(f2rspc, f2rs, TT, pc, q)
    moh!(mheads, f2rs, V, L, H, q)
    moh!(mheadspc, f2rspc, V, L, H, q)
    
end
    

    
    
    





    
    