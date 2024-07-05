struct Chain{T,Ti}
    seq::Array{Ti,1}
    log_prob::Array{T,1}
    sites::Array{Int,1}
    L::Int
    generator::Xoshiro
end

function Chain(seq::Array{<:Integer,1}, q::Int, generator::Xoshiro; T::DataType=Float32, Ti::DataType=Int8)
    L = size(seq,1)
    log_prob = T.(zeros(q))
    sites = [i for i in 1:L]
    seq = Ti.(seq)
    Chain{T,Ti}(seq, log_prob, sites, L, generator)
end


struct Data{T, T2}
    msa::Array{T2,2}
    f1::Array{T,1}
    f2::Array{T,2}
    f1rs::Array{T,2}
    f2rs::Array{T,4}
    f2rs_disc::Array{T,4}
    mheads::Array{T,3}
    mheads_disc::Array{T,3}
end

function Data(filepath, V, pc::AbstractFloat, q::Int, H::Int; T::DataType=Float32, T2::DataType = Int8)
    
    msa = T2.(quickread(filepath)[1])
    L = size(msa,1)
    f1, f2, M = compute_weighted_frequencies(msa, q+1, 0.2)
    f1 = T.(f1)
    f2 = T.(f2)
    f1rs = reshape(f1, (q, L))
    f2rs = reshape(f2, (q, L, q, L))
    f2rs_disc = reshape(f1*f1', (q, L, q, L))
    pseudocount1!(f1rs, f1rs, T, pc, q)
    pseudocount2!(f2rs, f2rs, T, pc, q)
    pseudocount2!(f2rs_disc, f2rs_disc, T, pc, q)
    mheads = T.(zeros(L,L,size(V,3)))
    mheads_disc = T.(zeros(L,L,size(V,3)))
    moh!(mheads, f2rs, V, L, H, q)
    moh!(mheads_disc, f2rs_disc, V, L, H, q)    
    
    Data{T, T2}(msa, f1, f2, f1rs, f2rs, f2rs_disc, mheads, mheads_disc)
end


struct ModelData{T, T2}
    msa::Array{T2,2}
    f1::Array{T,1}
    f2::Array{T,2}
    f1rs::Array{T,2}
    f2rs::Array{T,4}
    f2rs_disc::Array{T,4}
    mheads::Array{T,3}
    mheads_disc::Array{T,3}
end

function ModelData(chains, V, N_chains::Int, L::Int, pc::AbstractFloat, q::Int, H::Int; T::DataType=Float32, T2::DataType = Int8)
    
    msa = create_msa(chains, L, N_chains)
    f1, f2, M = compute_weighted_frequencies(msa, q+1, 0.2)
    f1 = T.(f1)
    f2 = T.(f2)
    f1rs = reshape(f1, (q, L))
    f2rs = reshape(f2, (q, L, q, L))
    f2rs_disc = reshape(f1*f1', (q, L, q, L))
    pseudocount1!(f1rs, f1rs, T, pc, q)
    pseudocount2!(f2rs, f2rs, T, pc, q)
    pseudocount2!(f2rs_disc, f2rs_disc, T, pc, q)
    mheads = T.(zeros(L,L,size(V,3)))
    mheads_disc = T.(zeros(L,L,size(V,3)))
    moh!(mheads, f2rs, V, L, H, q)
    moh!(mheads_disc, f2rs_disc, V, L, H, q)    
    
    Data{T, T2}(msa, f1, f2, f1rs, f2rs, f2rs_disc, mheads, mheads_disc)
end

function update_ModelData!(model, V::Array{T,3}, L::Int, pc::T, q::Int, H::Int) where {T}
    
    f1, f2 = compute_freq(Int8.(model.msa))
    model.f1 .= T.(f1)
    model.f2 .= T.(f2)
    model.f1rs .= reshape(f1, (q, L))
    model.f2rs .= reshape(f2, (q, L, q, L))
    model.f2rs_disc .= reshape(f1*f1', (q, L, q, L))
    pseudocount1!(model.f1rs, model.f1rs, T, pc, q)
    pseudocount2!(model.f2rs, model.f2rs, T, pc, q)
    pseudocount2!(model.f2rs_disc, model.f2rs_disc, T, pc, q)
    moh!(model.mheads, model.f2rs, V, L, H, q)
    moh!(model.mheads_disc, model.f2rs_disc, V, L, H, q)    
end



struct NumSolVar{T}
    k::Array{T,1}
    dest::Array{T,1}
end

function NumSolVar(T::DataType=Float32)
    k = T.(zeros(3))
    dest = T.(zeros(3))
    NumSolVar{T}(k, dest)
end


struct DeltaEnergy{T}
    pars::Array{T,1}
    deltahm::Array{T,1}
    deltahn::Array{T,1}
    k::T
end

function DeltaEnergy(q::Int, T::DataType=Float32)
    pars = T.(zeros(2*q+1))
    deltahm = T.(zeros(q))
    deltahn = T.(zeros(q))
    k = one(T)
    DeltaEnergy{T}(pars, deltahm, deltahn, k)
end



    
    