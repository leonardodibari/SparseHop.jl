

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


function prob_cond!(chain, 
        site::Int, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int,
        full_graf;
        q = 21)  where {T}

    
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

function fully_connected_graph(n)
    g = Graph(n)
    for i in 1:n
        for j in (i+1):n
            add_edge!(g, i, j)
        end
    end
    return g
end


        
                
 #look regularization    
function parallel_MCMC(V::Array{T,3}; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_chains = 1000, N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 0.05, 
        each_step = 10, lambda = 0.001, n_edges = 30, reg = 0.01, approx = false, verbose = false,
        opt_k = true, grad_upd = true) where {T}
    
    TT = eltype(V)
    H = size(V,3)
    D = Data(msa_file, V, T = TT)
    W = compute_weights(D.msa, 0.2)[1]
    _w = TT.(W ./ sum(W))
    L = size(D.msa,1)
    
    res = []
    x_res = []
    
    dL = zeros(L,L,H)
    k = zeros(L,L,H)
    y_k = zeros(L,L,H)
    
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
    f2rspc = TT.(pseudocount2(f2rs, pc = 0.1))
    mheads = TT.(zeros(L,L,size(V,3)))
    moh!(mheads, f2rspc, V)
    
    J = TT.(rand(21, L, 21, L))

    for iter in 1:N_iter
        
        #from time to time print info on learning
        if iter % each_step == 0
            get_J!(J, K, V)
            s = score(K,V)
            PPV = compute_PPV(s,structfile)
            println("Iter $(iter) One $(round(cor(f1[:],D.f1[:]), digits = 3)) Conn $(round(cor(triu(f2 - f1*f1', 21)[:], triu(D.f2 - D.f1*D.f1', 21)[:]), digits = 3)) PPV@L $(round(PPV[L], digits = 3)) PPV@2L $(round(PPV[2*L], digits = 3)) #edges $(sum([ne(graf[head]) for head in 1:H])) / $(Int(L*(L-1)*H/2)) par_ratio = $(round((L*21+ 21*21 + sum([ne(graf[head]) for head in 1:H]))/potts_par, digits = 3)) ") 
            println("Loss : $(get_loss_J(J, h, D.msa, _w))")
        end       
        
        start_edge = time()
        if approx == true && n_edges !== 0
            @tasks for i in 1:L 
                for j in i+1:L 
                    for h in 1:H 
                        k[i,j,h], dL[i,j,h] = dlog_small_k(D.f2rspc[:,i,:,j], f2rspc[:,i,:,j], V[:,:,h], reg = reg) 
                    end 
                end 
            end
            dL[K .!= 0] .= 0
            close("all"); plt.scatter(k[:],y_k[:]); savefig("../prova_k_y_k.png")
            close("all"); plt.scatter(k[:],dL[:]); savefig("../prova_k_dL.png")
            println("Suggested K : $(maximum(k)), terms $(argmax(dL)), dL : $(maximum(dL))")
            for _ in 1:n_edges
                m, n, nu = Tuple(argmax(dL))
                if opt_k == true
                    K[m,n,nu] = SparseHop.bisection(D.f2rspc[:,m,:,n], f2rspc[:,m,:,n], V[:,:,nu], reg = reg)[1]; 
                    K[n,m,nu] = K[m,n,nu]
                else
                    K[m,n,nu] = 0
                    K[n,m,nu] = K[m,n,nu]
                end
                add_edge!(graf[nu], m, n)
                add_edge!(full_graf, m, n)
                if verbose == true
                    print("Iteration : $(iter) ")
                    println((m, n, nu))
                    println((K[m,n,nu], dL_a[m,n,nu]))
                end
                dL[m,n,nu] = 0
            end
        end
        
        if approx == false && n_edges !== 0
            @tasks for i in 1:L 
                for j in i+1:L 
                    for h in 1:H 
                        k[i,j,h], y_k[i,j,h] = bisection(D.f2rspc[:,i,:,j], f2rspc[:,i,:,j], V[:,:,h], reg = reg)
                        dL[i,j,h] = dlog(D.f2rspc[:,i,:,j], f2rspc[:,i,:,j], k[i,j,h], V[:,:,h], reg = reg)
                    end 
                end 
            end
            dL[K .!= 0] .= 0
            #close("all"); plt.scatter(k[:],y_k[:]); savefig("../prova_k_y_k.png")
            #close("all"); plt.scatter(k[:],dL[:]); savefig("../prova_k_dL.png")
            println("Suggested K : $(maximum(k)), terms $(argmax(dL)), dL : $(maximum(dL))")
            for _ in 1:n_edges
                m, n, nu = Tuple(argmax(dL))
                
                if opt_k == true
                    K[m,n,nu] = k[m,n,nu] 
                    K[n,m,nu] = K[m,n,nu]
                else
                    K[m,n,nu] = 0
                    K[n,m,nu] = K[m,n,nu]
                end
                add_edge!(graf[nu], m, n)
                add_edge!(full_graf, m, n)
                if verbose == true
                    print("Iteration : $(iter) ")
                    println((m, n, nu))
                    println((K[m,n,nu], dL[m,n,nu]))
                end
                dL[m,n,nu] = 0
            end
        end
        fine_edge = time()
        #println("Edge act. $(n_edges) edges took $(fine_edge-start_edge)") 
        
        
        start_gibbs = time()
        for _ in 1:grad_iter
            start = time()
            get_J!(J, K, V)
            @tasks for n in 1:N_chains
                gibbs_sampling!(chains[n], h, J, L, full_graf, sweeps = sweeps)
                for i in 1:L
                    msa[i,n] = chains[n].seq[i]
                end
            end
            fine = time()
            #println("$(sweeps) sweeps take $(fine-start) ") 
            
        
            update_sample!(msa, f1, f2, f1rs, f2rs, f2rspc, V, mheads, L, TT)
            push!(x_res, maximum(K))
            if grad_upd == true
                #gradient descent on fields
                h .+= learn_r .* (D.f1rs .- f1rs) 
                #gradient descent on couplings only for selected graf
                for head in 1:H
                    if ne(graf[head]) !== 0
                        for i in findall(.!isempty.(graf[head].fadjlist))
                            for j in neighbors(graf[head], i)
                                K[i, j, head] += learn_r * (D.mheads[i,j,head] - mheads[i,j,head])
                                push!(res, D.mheads[i,j,head] - mheads[i,j,head])
                                #K[i, j, head] += 0.015
                                #println("Grad $(D.mheads[i,j,head] - mheads[i,j,head])  Coup: $(K[i,j,head]) $(i) $(j) $(head)")
                            end
                        end
                    end
                end            
                
            end   
            fine_gibbs = time()
            #println("Gibbs sampl. $(sweeps) sweeps $(grad_iter) grad up took $(fine_gibbs-start_gibbs)")
        end
    end
    
    return (K = K, h = h, g = graf, Z = msa, Zf = D.msa, C = chains, 
        f1 = f1rs, Df1 = D.f1rs, f2 = f2rs, Df2 = D.f2rs, res = res, x_res = x_res)
end


