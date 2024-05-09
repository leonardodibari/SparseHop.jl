function prob_cond!(chain, 
        site::Int, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int,
        full_graf) where {T}

	for a in 1:21
        chain.log_prob[a] = 0
		chain.log_prob[a] += h[a, site]
 		for j in neighbors(full_graf, site)
			chain.log_prob[a] += J[chain.seq[j], j, a, site]
        end
	end
    loc_softmax!(chain.log_prob)
end


function gibbs_sampling!(chain, h::Array{T,2}, J::Array{T,4}, L::Int, graf, sweeps::Int) where {T}
    for s in 1:sweeps
        for site in shuffle!(chain.sites)
            prob_cond!(chain, site, h, J, L, graf)
            loc_sample!(chain.generator, chain.log_prob, chain.seq, site)
        end
    end
end


function fully_connected_graph(n)
    g = Graph(n)
    for i in 1:n
        for j in (i+1):n
            add_edge!(g, i, j)
        end
    end
    return g
end


function single_moh(f::Array{T, 4}, V::Array{T,3}, i::Int, j::Int, head::Int) where {T}
    @tullio media := f[a, i, b, j] * V[a, b, head] * (i != j)
    return media
end

function moh!(dest::Array{T,3}, f::Array{T, 4}, V::Array{T,3}, L::Int, H::Int, q::Int) where {T}
    dest .= 0
    @inbounds for i in 1:L
        @inbounds for j in 1:L
            @inbounds for head in 1:H
                for a in 1:q
                    for b in 1:q
                        dest[i, j, head] += f[a, i, b, j] * V[a, b, head] * (i != j)
                    end
                end
            end
        end
    end
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
    TT::DataType) where {T}
    
    f1, f2 = compute_freq(Int8.(msa))
    f1 .= TT.(f1)
    f2 .= TT.(f2)
    f1rs .= reshape(f1, (q, L))
    f2rs .= reshape(f2, (q, L, q, L))
    pseudocount2!(f2rspc, f2rs, TT, pc, q)
    moh!(mheads, f2rs, V, L, H, q)
    moh!(mheadspc, f2rspc, V, L, H, q)
    
end
    


function get_loss_J(J::Array{T,4},
    h::Array{T,2},
    Z::Array{Int8,2}, 
    _w::Array{T, 1}) where {T}
    
    @tullio en0[a, i, m] := J[a, i, Z[j, m], j];
    en = en0 .+ h 
    @tullio data_en[i, m] := en[Z[i, m], i, m] 
    log_z = dropdims(logsumexp(en, dims=1), dims=1)
    @tullio loss[i] := _w[m]*(log_z[i, m] - data_en[i,m])
  
    return sum(loss) 
end
