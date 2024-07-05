function prob_cond!(chain, 
        site::Int, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int,
        full_graf) where {T}
    
    fill!(chain.log_prob, T(0))
	@inbounds for a in 1:21
		chain.log_prob[a] += h[a, site]
 		for j in neighbors(full_graf, site)
			chain.log_prob[a] += J[chain.seq[j], j, a, site]
        end
	end
    loc_softmax!(chain.log_prob)
end

function prob_cond2!(chain, 
        site::Int, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int,
        full_graf) where {T}
    
    ne = neighbors(full_graf, site)
    fill!(chain.log_prob, T(0))
    @inbounds for a in 1:21
        chain.log_prob[a] += h[a, site]
        for j in ne
            chain.log_prob[a] += J[chain.seq[j], j, a, site]
        end
    end	
    #loc_softmax!(chain.log_prob)
end


function gibbs_sampling!(chain, h::Array{T,2}, J::Array{T,4}, L::Int, full_graf, sweeps::Int) where {T}
    @inbounds for s in 1:sweeps
        for site in shuffle!(chain.sites)
            prob_cond!(chain, site, h, J, L, full_graf)
            loc_sample!(chain.generator, chain.log_prob, chain.seq, site)
        end
    end
end


function run_gibbs_sampling!(chains, msa::Array{Int8,2}, h::Array{T,2}, J::Array{T,4}, K::Array{T,3}, V::Array{T,3},L::Int, full_graf, sweeps::Int, N_chains::Int) where {T}
    
    get_J!(J, K, V)
    @tasks for n in 1:N_chains
        gibbs_sampling!(chains[n], h, J, L, full_graf, sweeps)
        for i in 1:L
           msa[i,n] = chains[n].seq[i]
        end
    end
    
end


function new_prob_cond!(chain, 
        site::Int, 
        h::Array{T,2}, 
        K::Array{T,3},
        V::Array{T,3},
        L::Int,
        H::Int,
        graf) where {T}

	@inbounds for a in 1:21
        chain.log_prob[a] = T(0)
		chain.log_prob[a] += h[a, site]
        for head in 1:H
            for j in neighbors(graf[head], site)
                chain.log_prob[a] += K[j,site,head]*V[chain.seq[j],a,head]
            end
        end
	end
    #loc_softmax!(chain.log_prob)
end


function new_gibbs_sampling!(chain, h::Array{T,2}, K::Array{T,3},
        V::Array{T,3}, L::Int, graf, sweeps::Int, H::Int) where {T}
    @inbounds for s in 1:sweeps
        for site in shuffle!(chain.sites)
            new_prob_cond!(chain, site, h, K, V, L, H, graf)
            loc_sample!(chain.generator, chain.log_prob, chain.seq, site)
        end
    end
end


function new_run_gibbs_sampling!(chains, msa::Array{Int8,2}, h::Array{T,2}, K::Array{T,3}, V::Array{T,3},L::Int, graf, sweeps::Int, N_chains::Int, H::Int) where {T}

    @tasks for n in 1:N_chains
        new_gibbs_sampling!(chains[n], h, K, V, L, graf, sweeps, H)
        for i in 1:L
            msa[i,n] = chains[n].seq[i]
        end
    end
    
end

 
function create_msa(chains, L::Int, N_chains::Int)
    msa = Int8.(zeros(L, N_chains))
    @tasks for n in 1:N_chains
        for i in 1:L
            msa[i,n] = chains[n].seq[i]
        end
    end
    return msa
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
    for i in eachindex(dest)
        dest[i] = zero(T)
    end
    
    @inbounds for i in 1:L
        @inbounds for j in (i+1):L
            @inbounds for head in 1:H
                for a in 1:q
                    for b in 1:q
                        dest[i, j, head] += f[a, i, b, j] * V[a, b, head] * (i != j) 
                    end
                end
                dest[j, i, head] += dest[i, j, head]
            end
        end
    end
end


function partition_f_one_edge(h::Array{T,2}, J::Array{T,4}, L::Int, i::Int, j::Int, q::Int) where {T}
    tot_z = zero(T)  
    for a in 1:q
        for b in 1:q
            tot_z+=exp(h[a,i]+h[b,j]+J[a,i,b,j])
        end
    end 
    return tot_z
end


function analytic_moh(h::Array{T,2}, J::Array{T,4}, V::Array{T,3}, z_tot::T, 
        L::Int, i::Int, j::Int, head::Int, q::Int) where {T}
    res = zero(T)
    for a in 1:q
        for b in 1:q
            p = exp(h[a,i] + h[b,j] + J[a,i,b,j])/z_tot
            res += p*V[a,b,head]
        end
    end
    return res 
end


function imp_moh(k::T, f2::Array{T,4}, V::Array{T,3},
        L::Int, i::Int, j::Int, head::Int, q::Int) where {T}
    num = zero(T)
    den = zero(T)
    for a in 1:q 
        for b in 1:q
            num += exp(k * V[a,b,head])*V[a,b,head]*f2[a,i,b,j]
            den += exp(k * V[a,b,head])*f2[a,i,b,j]
        end
    end
    return num/den 
end

function energy(seq::Array{Int8}, h::Array{T,2}, J::Array{T,4}, L::Int) where {T}
    res = zero(T)
    for i in 1:L
        res += h[seq[i],i]
        for j in i+1:L
            res += J[seq[i], i, seq[j], j]
        end
    end
    return res
end
                 
