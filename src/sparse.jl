

function single_moh(f::Array{T, 4}, V::Array{T,3}, i::Int, j::Int, head::Int) where {T}
    @tullio media := f[a, i, b, j] * V[a, b, head] * (i != j)
    return media
end

function moh!(dest::Array{T,3}, f::Array{T, 4}, V::Array{T,3}) where {T}
    @tullio dest[i, j, h] = f[a, i, b, j] * V[a, b, h] * (i != j)
end

function soh!(dest::Array{T,3}, f::Array{T, 4}, V::Array{T,3}) where {T}
    @tullio dest[i, j, h] = f[a, i, b, j] * V[a, b, h] * V[a, b, h] 
end


function small_k_single_edge(f::Array{T, 4}, f_emp::Array{T, 4}, 
        V::Array{T,3}, i::Int, j::Int, head::Int) where {T}
    @tullio media := f[a, i, b, j] * V[a, b, head] * (i != j)
    @tullio media_emp := f_emp[a, i, b, j] * V[a, b, head] * (i != j)
    @tullio square := f[a, i, b, j] * V[a, b, head] * V[a, b, head]  * (i != j)  
    return (media_emp - media) / (square - media^2)
end

function small_k_all_edges(f::Array{T, 4}, f_emp::Array{T, 4}, 
        V::Array{T,3}) where {T}
    @tullio media[i, j, head] := f[a, i, b, j] * V[a, b, head] * (i != j)
    @tullio media_emp[i, j, head] := f_emp[a, i, b, j] * V[a, b, head] * (i != j)
    @tullio square[i, j, head] := f[a, i, b, j] * V[a, b, head] * V[a, b, head] * (i != j)  
    return (media_emp .-  media) ./ (square .- (media.^2))
end

function compute_z_score!(dest::Array{T,3}, f::Array{T, 4}, f_emp::Array{T, 4}, 
        V::Array{T,3}) where {T}
    @tullio media[i, j, head] := f[a, i, b, j] * V[a, b, head] * (i != j)
    @tullio media_emp[i, j, head] := f_emp[a, i, b, j] * V[a, b, head] * (i != j)
    @tullio square[i, j, head] := f[a, i, b, j] * V[a, b, head] * V[a, b, head]  * (i != j)  
    dest = ((media_emp .-  media).^2) ./ (square .- (media.^2))
end


function compute_z_score(f::Array{T, 4}, f_emp::Array{T, 4}, 
        V::Array{T,3}) where {T}
    @tullio media[i, j, head] := f[a, i, b, j] * V[a, b, head] * (i != j)
    @tullio media_emp[i, j, head] := f_emp[a, i, b, j] * V[a, b, head] * (i != j)
    @tullio square[i, j, head] := f[a, i, b, j] * V[a, b, head] * V[a, b, head]  * (i != j)  
    return ((media_emp .-  media).^2) ./ (square .- (media.^2))
end


function compute_z_score(mheads::Array{T,3}, sqheads::Array{T,3}, D) where {T}
    num = (D.mheads .- mheads) .^2
    den = sqheads .- (mheads .^2)
    arr = num ./ den
    a = argmax(arr)[1]
    b = argmax(arr)[2]
    c = argmax(arr)[3]
    #println((a ,b,c, arr[a,b,c], num[a,b,c], den[a,b,c]))
    return ((D.mheads .- mheads) .^2) ./ (sqheads .- (mheads .^2))
end


function edge_act!(z_score::Array{T, 3}, K::Array{T,3}, V::Array{T,3}, f2rs::Array{T,4},
        df2rs::Array{T,4}, graf, full_graf; n_edges = 30) where {T}
    z = deepcopy(z_score)
    for i in 1:n_edges
        m, n, nu = Tuple(argmax(z))
        add_edge!(graf[nu], m, n)
        add_edge!(full_graf, m, n)
        K[m, n, nu] = small_k_single_edge(pseudocount2(f2rs, pc = 0.1), pseudocount2(df2rs, pc = 0.1), V, m, n, nu)
        K[n, m, nu] = K[m, n, nu]
        z[m, n, nu] = 0
        z[n, m, nu] = 0
    end
end


function prob_cond!(chain, 
        site::Int, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int,
        full_graf;
        q = 21)  where {T}

    #log_prob = zeros(q)
	for a in 1:q
        chain.log_prob[a] = 0
		chain.log_prob[a] += h[a, site]
 		for j in neighbors(full_graf, site)
			chain.log_prob[a] += J[chain.seq[j], j, a, site]
        end
	end
    
    chain.prob .= softmax(chain.log_prob)
end


function gibbs_sampling!(chain, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int, 
        graf;
        q = 21, sweeps = 5) where {T}
    
    for _ in 1:sweeps
        for site in randperm(L)
            prob_cond!(chain, site, h, J, L, graf)
            chain.seq[site] = sample(chain.generator, 1:q, weights(chain.prob))
        end
    end
end


        
                
 #look regularization    
function parallel_MCMC(V::Array{T,3}; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_chains = 1000, N_iter::Int = 100, sweeps::Int = 5, learn_r = 0.05, 
        each_step = 10, lambda = 0.001, n_edges = 30, reg = 0.01) where {T}
    
    TT = eltype(V)
    H = size(V,3)
    D = Data(msa_file, V, T = TT)
    L = size(D.msa,1)
    
    dL_a = zeros(L,L,H)
    k_a = zeros(L,L,H)
    
    rng = random_gens(N_chains)
    chains = [Chain(Int8.(rand(1:21, L)), rng[n]) for n in 1:N_chains] #initialize random
    #chains = [Chain(D.msa[:,n], rng[n]) for n in 1:N_chains]   #initialize in the msa
    graf = [Graph(L) for head in 1:H]
    full_graf = Graph(L)

    
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
    #mheads = TT.(zeros(L,L,size(V,3)))
    #moh!(mheads, pseudocount2(f2rs, pc = 0.1), V)
    #sqheads = TT.(zeros(L,L,size(V,3)))
    #soh!(sqheads, pseudocount2(f2rs, pc = 0.1), V)
    
    
    J = TT.(rand(21, L, 21, L))
    
    for iter in 1:N_iter
        
        #from time to time print info on learning
        if iter % each_step == 0
            s = score(K,V)
            PPV = compute_PPV(s,structfile)
            println("Iter $(iter) One $(round(cor(f1[:],D.f1[:]), digits = 3)) Conn $(round(cor(triu(f2 - f1*f1', 21)[:], triu(D.f2 - D.f1*D.f1', 21)[:]), digits = 3)) PPV@L $(round(PPV[L], digits = 3)) PPV@2L $(round(PPV[2*L], digits = 3)) #edges $(sum([ne(graf[head]) for head in 1:H])) / $(Int(L*(L-1)*H/2)) par_ratio = $(round((L*21+ 21*21 + sum([ne(graf[head]) for head in 1:H]))/potts_par, digits = 3)) ")     
        end       
        
         
        @tasks for i in 1:L 
            for j in i+1:L 
                for h in 1:H 
                   k_a[i,j,h], dL_a[i,j,h] = dlog_small_k(pseudocount2(D.f2rs[:,i,:,j], pc = 0.1), 
                        pseudocount2(f2rs[:,i,:,j], pc = 0.1), V[:,:,h], reg = reg); 
                end 
            end 
        end
        
        for _ in 1:n_edges
            m, n, nu = Tuple(argmax(dL_a))
            #K[m,n,nu] = SparseHop.bisection(pseudocount2(D.f2rs[:,m,:,n], pc = 0.1), 
             #   pseudocount2(f2rs[:,m,:,n], pc = 0.1), V[:,:,nu], reg = reg)[1]; 
            #K[n,m,nu] = K[m,n,nu]
        
        
            add_edge!(graf[nu], m, n)
            add_edge!(full_graf, m, n)
        
            print("Iteration : $(iter) ")
            println((m, n, nu))
            println((K[m,n,nu], dL_a[m,n,nu]))
            dL_a[m,n,nu] = 0
        end
        
        
        
        get_J!(J, K, V)
        for n in 1:N_chains
            gibbs_sampling!(chains[n], h, J, L, full_graf, sweeps = sweeps)
            for i in 1:L
                msa[i,n] = chains[n].seq[i]
            end
        end
        
        
        #println(mean([get_energy(Smsa[:,n], K, V, h) for n in 1:N_chains]))
        #update the model sample frequencies 
        f1, f2 = compute_freq(Int8.(msa))
        f1 = TT.(f1)
        f2 = TT.(f2)
        f1rs = reshape(f1, (21, L))
        f2rs = reshape(f2, (21, L, 21, L))
        #moh!(mheads, pseudocount2(f2rs, pc = 0.1), V)
        #soh!(sqheads, pseudocount2(f2rs, pc = 0.1), V)
        #update_sample!(msa, f1, f2, f1rs, f2rs,  V, mheads, sqheads,chains, L, N_chains)
        
        
        
        #gradient descent on fields
        h .+= learn_r .* (D.f1rs .- f1rs) 
        #gradient descent on couplings only for selected graf
        for head in 1:H
            if ne(graf[head]) !== 0
                for i in findall(.!isempty.(graf[head].fadjlist))
                    for j in neighbors(graf[head], i)
                        
                        K[i, j, head] += learn_r * (single_moh(D.f2rs, V, i, j, head) 
                            .- single_moh(f2rs, V, i, j, head))
                        
                    end
                end
            end
        end
         
        
    end
    
    return (K = K, h = h, g = graf, Z = msa, Zf = D.msa, C = chains, #dL = z_score, 
        f1 = f1rs, Df1 = D.f1rs, f2 = f2rs, Df2 = D.f2rs)
end


