



function prob_cond_ns!(chain, 
        site::Int, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int;
        q = 21)  where {T}

    
	for a in 1:q
        chain.log_prob[a] = 0
		chain.log_prob[a] += h[a, site]
 		for j in 1:L
			chain.log_prob[a] += J[chain.seq[j], j, a, site]
        end
	end
    
    chain.prob .= softmax(chain.log_prob)
end


function gibbs_sampling_ns!(chain, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int;
        q = 21, sweeps = 5) where {T}
    
    for _ in 1:sweeps
        for site in randperm(L)
            prob_cond_ns!(chain, site, h, J, L)
            chain.seq[site] = sample(chain.generator, 1:q, weights(chain.prob))
        end
    end
end




        
                
 #look regularization    
function parallel_MCMC_ns(V::Array{T,3}; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_chains = 1000, N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 0.05, 
        each_step = 10, lambda = 0.001, n_edges = 30, reg = 0.01, approx = false, verbose = false) where {T}
    
    TT = eltype(V)
    H = size(V,3)
    D = Data(msa_file, V, T = TT)
    W = compute_weights(D.msa, 0.2)[1]
    _w = TT.(W ./ sum(W))
    L = size(D.msa,1)
    
    dL = zeros(L,L,H)
    k = zeros(L,L,H)
    y_k = zeros(L,L,H)
    
    rng = random_gens(N_chains)
    chains = [Chain(Int8.(rand(1:21, L)), rng[n]) for n in 1:N_chains] #initialize random
    #chains = [Chain(D.msa[:,n], rng[n]) for n in 1:N_chains]   #initialize in the msa
    
    
    potts_par = 21*21*L*(L-1)/2 + L*21

    K = TT.(zeros(L,L,H))
    h = TT.(log.(pseudocount1(D.f1rs, pc = 0.0001)))
    
    msa = Int8.(zeros(L, N_chains))
    @tasks for n in 1:N_chains
        for i in 1:L
            msa[i,n] = chains[n].seq[i]
        end
    end
      
    f1, f2 = compute_freq(Int8.(msa))
    f1 = TT.(f1)
    f2 = TT.(f2)
    f1rs = reshape(f1, (21, L))
    f2rs = reshape(f2, (21, L, 21, L))
    mheads = TT.(zeros(L,L,H))
    moh!(mheads, pseudocount2(f2rs, pc = 0.1), V)
    #sqheads = TT.(zeros(L,L,size(V,3)))
    #soh!(sqheads, pseudocount2(f2rs, pc = 0.1), V)
    
    
    J = TT.(rand(21, L, 21, L))

    for iter in 1:N_iter
        
        #from time to time print info on learning
        if iter % each_step == 0
            s = score(K,V)
            PPV = compute_PPV(s,structfile)
            println("Iter $(iter) One $(round(cor(f1[:],D.f1[:]), digits = 3)) Conn $(round(cor(triu(f2 - f1*f1', 21)[:], triu(D.f2 - D.f1*D.f1', 21)[:]), digits = 3)) PPV@L $(round(PPV[L], digits = 3)) PPV@2L $(round(PPV[2*L], digits = 3)) ")     
        end       
      
        
        for _ in 1:grad_iter 
            get_J!(J, K, V)
            for n in 1:N_chains
                gibbs_sampling_ns!(chains[n], h, J, L, sweeps = sweeps)
                for i in 1:L
                    msa[i,n] = chains[n].seq[i]
                end
            end
        
            #update the model sample frequencies 
            f1, f2 = compute_freq(Int8.(msa))
            f1 .= TT.(f1)
            f2 .= TT.(f2)
            f1rs .= reshape(f1, (21, L))
            f2rs .= reshape(f2, (21, L, 21, L))
            moh!(mheads, pseudocount2(f2rs, pc = 0.1), V)
            #soh!(sqheads, pseudocount2(f2rs, pc = 0.1), V)
            #update_sample!(msa, f1, f2, f1rs, f2rs,  V, mheads, sqheads,chains, L, N_chains)
        
            #gradient descent on fields
            #h .+= learn_r .* (D.f1rs .- f1rs) 
            #gradient descent on couplings only for selected graf
            K .+= learn_r .* (D.mheads .- mheads)
            
            println("Loss : $(get_loss_J(J, h, D.msa, _w))")
        end        
    end
    
    return (K = K, h = h, Z = msa, Zf = D.msa, C = chains,  
        f1 = f1rs, Df1 = D.f1rs, f2 = f2rs, Df2 = D.f2rs)
end


