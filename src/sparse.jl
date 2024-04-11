function prob_cond(site, 
        seq::Array{<:Integer,1}, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int,
        graf;
        q = 21, t = 1)  where {T}

	prob = zeros(q)
    log_prob = zeros(q)
    ne = neighbors(graf, site)
    
	for a in 1:q
		log_prob[a] += h[a, site]
 		for j in ne
			log_prob[a] += J[seq[j], j, a, site]
        end
	end
 
	return softmax_notinplace(log_prob)
end


function gibbs_sampling!(seq::Array{<:Integer,1}, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int, 
        graf,
        g;
        q = 21, sweeps = 5) where {T}
    
    for n in 1:sweeps
        for i in 1:L
            seq[i] = sample(g, 1:q, weights(prob_cond(i, seq, h, J, L, graf)))
        end
    end
    return seq
end

function single_diff_moh(f::Array{Float64, 4}, f_emp::Array{Float64, 4}, 
        V::Array{T,3}, i::Int, j::Int, head::Int) where {T}
    
    @tullio media := f[a, i, b, j] * V[a, b, head]
    @tullio media_emp := f_emp[a, i, b, j] * V[a, b, head]
    return media_emp .- media
end

function diff_moh(f::Array{Float64, 4}, f_emp::Array{Float64, 4}, V::Array{T,3}) where {T}
    return moh(f_emp, V) .- moh(f, V)
end

function moh(f::Array{Float64, 4}, V::Array{T,3}) where {T}
    @tullio media[i, j, h] := f[a, i, b, j] * V[a, b, h]
    return media
end

function voh(f::Array{Float64, 4}, V::Array{T,3}) where {T}
    @tullio square[i, j, h] := f[a, i, b, j] * V[a, b, h] * V[a, b, h] 
    return square .- (moh(f, V) .* 2)
end

function compute_z_score(f::Array{Float64, 4}, f_emp::Array{Float64, 4}, V::Array{T,3}) where {T}
    return diff_moh(f, f_emp, V) ./ voh(f, V)
end


function find_optimal_K(f2::Array{Float64, 4}, f2_emp::Array{Float64, 4}, V::Array{T,3}, 
        m::Int, n::Int, nu::Int) where {T}
    println("ciao")
end
                
                
    
function parallel_MCMC(chains, 
        KK::Array{T,3},
        V::Array{T,3},
        Z_fam::Array{<:Integer,2}; 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", N_iter::Int = 100, 
        sweeps::Int = 5, learn_r = 0.05, each_step = 10, lambda = 0.001, n_edges = 30) where {T}
    
    L = chains[1].L
    H = size(V,3)
    K = deepcopy(KK)
    TT = eltype(V)
    graf = [Graph(L) for head in 1:H]
    N_chains = size(chains,1)
    f1_emp, f2_emp, M = compute_weighted_frequencies(Z_fam, 22, 0.2)
    Z = zeros(L, N_chains)
    h = TT.(log.( pseudocount1(reshape(f1_emp, (21, L)), pc = 0.001)))
    
    for n in 1:N_chains
        Z[:,n] = chains[n].seq
    end
    f1, f2 = compute_freq(Int8.(Z))
    println("START: One $(round(cor(f1[:],f1_emp[:]), digits = 3)) Two $(round(cor(f2[:],f2_emp[:]), digits = 3))")
    
    
    for iter in 1:N_iter
        J = get_J(K, V)
        @tasks for n in 1:N_chains
            Z[:,n] = gibbs_sampling!(chains[n].seq, h, J, L, graf[1], chains[n].generator, sweeps = sweeps)
        end

        if iter % each_step == 0
            for n in 1:N_chains
                Z[:,n] = chains[n].seq
            end
            f1, f2 = compute_freq(Int8.(Z))
            s = score(K,V)
            PPV = compute_PPV(s,structfile)
            println("Iter $(iter) One $(round(cor(f1[:],f1_emp[:]), digits = 3)) Two $(round(cor(f2[:],f2_emp[:]), digits = 3)) PPV@L $(round(PPV[L], digits = 3)) PPV@2L $(round(PPV[2*L], digits = 3))")
            println("Num of edges = $(sum([ne(graf[head]) for head in 1:H])) / $(L*(L-1)*H/2) ")     
        end       
    
        f1, f2 = compute_freq(Int8.(Z))
        f1_diff = reshape(f1_emp .- f1, (21, L))
        f2_emp = reshape(f2_emp, (21, L, 21, L))
        f2 = reshape(f2, (21, L, 21, L))
        
        #gradient descent on fields
        h .+= (learn_r .* f1_diff) .+  2 .* lambda .* h
        
        #gradient descent on couplings only for selected graf
        for head in 1:H
            if ne(graf[head]) !== 0
                for i in findall(.!isempty.(graf[head].fadjlist))
                    for j in neighbors(graf[head], i)
                        #print("Iteration : $(iter) ")
                        #println((i, j, head))
                        K[i,j,head] += learn_r * single_diff_moh(f2, f2_emp, V, i, j, head) 
                                        + 2 * lambda * K[i, j, head]
                        #println(K[i,j,head])
                    end
                end
            end
        end
                     
        #edge activation/update
        z_score = compute_z_score(pseudocount2(f2), pseudocount2(f2_emp), V)
        edge_act!(z_score, graf, n_edges = n_edges)        
       
    end

    return (K = K, h = h)
end


