   
function parallel_MCMC(V::Array{T,3}; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_chains = 1000, N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 0.05, 
        each_step = 10, q = 21, pc = 0.1, n_edges = 30, reg = 0.01, avoid_upd = false, verbose = false, opt_k = true, grad_upd = true) where {T}
    
    TT = eltype(V)
    H = size(V,3)
    pc = TT(pc)
    D = Data(msa_file, V, pc, q, H, T = TT)
    W = compute_weights(D.msa, 0.2)[1]
    _w = TT.(W ./ sum(W))
    
    L = size(D.msa,1)
     
    dL = zeros(L,L,H)
    k = zeros(L,L,H)
    y_k = zeros(L,L,H)
    
    rng = random_gens(N_chains)
    chains = [Chain(Int8.(rand(1:21, L)), q, rng[n]) for n in 1:N_chains] #initialize random
    #chains = [Chain(D.msa[:,n], rng[n]) for n in 1:N_chains]   #initialize in the msa
    graf = [Graph(L) for head in 1:H]
    full_graf = Graph(L)
    str = [NumSolVar(TT) for head in 1:H]
    
    potts_par = q*q*L*(L-1)/2 + L*q

    K = TT.(zeros(L,L,H)) 
    h = TT.(log.(pseudocount1(D.f1rs, TT, 0.0001, q)))
    
    msa = Int8.(zeros(L, N_chains))
    @tasks for n in 1:N_chains
        for i in 1:L
            msa[i,n] = chains[n].seq[i]
        end
    end
  
    #initial thermalization of random sample to local fields with 20 gibbs sweeps
    J = TT.(rand(q, L, q, L))
    get_J!(J, K, V)
    @tasks for n in 1:N_chains
        gibbs_sampling!(chains[n], h, J, L, full_graf, 20)
        for i in 1:L
            msa[i,n] = chains[n].seq[i]
        end
    end
    
    res = []
    y_res = []
    
    f1, f2 = compute_freq(Int8.(msa))
    f1 = TT.(f1)
    f2 = TT.(f2)
    f1rs = reshape(f1, (q, L))
    f2rs = reshape(f2, (q, L, q, L))
    f2rspc = pseudocount2(f2rs, TT, pc, q)
    mheads = TT.(zeros(L,L,size(V,3)))
    moh!(mheads, f2rs, V, L, H, q)
    mheadspc = TT.(zeros(L,L,size(V,3)))
    moh!(mheadspc, f2rspc, V, L, H, q)
    
  
    for iter in 1:N_iter
        
        #from time to time print info on learning
        if iter % each_step == 0
            get_J!(J, K, V)
            s = score(K,V)
            PPV = compute_PPV(s,structfile)
            println("Iter $(iter) One $(round(cor(f1[:],D.f1[:]), digits = 3)) Conn $(round(cor(triu(f2 - f1*f1', 21)[:], triu(D.f2 - D.f1*D.f1', 21)[:]), digits = 3)) PPV@L $(round(PPV[L], digits = 3)) PPV@2L $(round(PPV[2*L], digits = 3)) #edges $(sum([ne(graf[head]) for head in 1:H])) / $(Int(L*(L-1)*H/2)) par_ratio = $(round((L*q+ q*H + sum([ne(graf[head]) for head in 1:H]))/potts_par, digits = 3)) ") 
            println("Loss : $(get_loss_J(J, h, D.msa, _w))")
        end       
        
        start = time()
        
        #edge activation
        if n_edges !== 0
            for i in 1:L 
                for j in 1:L 
                    if j != i 
                        @tasks for head in 1:H
                            bisection!(k, y_k, dL, str[head], D.mheadspc[i,j,head], f2rspc, V, i, j, head, reg, q)
                        end
                    end 
                end 
            end
            #close("all"); scatter(k[:],y_k[:], label ="k vs y_k $(iter)"); plt.legend(); savefig("../$(iter)Kvsy_Kreg$(reg).png");
            #close("all"); scatter(k[:],dL[:], label ="k vs dL $(iter)"); plt.legend(); savefig("../$(iter)KvsdLreg$(reg).png");
            #println("max numerical_error : $(maximum(abs.(y_k)))")
            if avoid_upd == true
                dL[K .!= 0] .= 0
            end
            
            for _ in 1:n_edges
                m, n, nu = Tuple(argmax(dL))
                if verbose == true
                    println("Suggested K : $(k[m,n,nu]), terms $(argmax(dL)), dL : $(maximum(dL))")
                end
                if opt_k == true
                    K[m,n,nu] = k[m,n,nu] 
                    K[n,m,nu] = K[m,n,nu]
                else
                    K[m,n,nu] = 0.
                    K[n,m,nu] = K[m,n,nu]
                end
                add_edge!(graf[nu], m, n)
                add_edge!(full_graf, m, n)
                dL[m,n,nu] = 0
            end
        end
        fine = time()
        if verbose == true
            println("Edge activation took $(fine-start) s")
        end
         
        startg = time()
        #gibbs sampling and gradient update
        for it in 1:grad_iter
            get_J!(J, K, V)
            
            @tasks for n in 1:N_chains
                gibbs_sampling!(chains[n], h, J, L, full_graf, sweeps)
                for i in 1:L
                    msa[i,n] = chains[n].seq[i]
                end
            end
        
            update_sample!(msa, f1, f2, f1rs, f2rs, f2rspc, V, mheads, mheadspc, L, pc, q, H, TT)
            
            if grad_upd == true
                #gradient descent on fields
                h .+= learn_r .* (D.f1rs .- f1rs)
                #gradient descent on couplings only for selected graf
                for head in 1:H
                    if ne(graf[head]) !== 0
                        for i in findall(.!isempty.(graf[head].fadjlist))
                            for j in neighbors(graf[head], i)
                                #K[i, j, head] += learn_r * (D.mheads[i,j,head] - mheads[i,j,head])
                                #println((i, j, head))
                                push!(res, K[i, j, head]) 
                                #print("Grad_it : $(it) K : $(K[i,j,head]) ")
                                K[i, j, head] += 0.01
                                push!(y_res, D.mheads[i,j,head] - mheads[i,j,head])
                                #println("Grad : $(D.mheads[i,j,head] - mheads[i,j,head]) ")                         
                            end
                        end
                    end
                end            
            end   
        end
        fineg = time()
        if verbose == true
            println("Sampling+grad update took $(fineg-startg) s")
        end
        
    end
    
    return (K = K, h = h, g = graf, Z = msa, Zf = D.msa, C = chains, 
        f1 = f1rs, Df1 = D.f1rs, f2 = f2rspc, Df2 = D.f2rspc, cost = mheadspc, res = res, y_res = y_res)
end


