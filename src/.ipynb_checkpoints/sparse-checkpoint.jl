
function runSparseHop(V::Array{T,3}; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_chains = 5000, N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 1e-2, 
        each_step = 10, q = 21, pc = 1e-2, n_edges = 30, reg = 1e-2, rand_init = true, avoid_upd = false, verbose = false, opt_k = true, grad_upd = true, savefile::Union{String, Nothing} = nothing) where {T}
    
    #global variables
    TT = eltype(V); pc = TT(pc); reg = TT(reg); learn_r = TT(learn_r); 
    H = size(V,3)
    D = Data(msa_file, V, pc, q, H, T = T)
    L,Mtot = size(D.msa) 
    
    #containers
    dL = TT.(zeros(L,L,H)); k = TT.(zeros(L,L,H)); y_k = TT.(zeros(L,L,H)); history = Int.(zeros(L,L,H)); 
    order_list = [];
   
    #structures & types
    rng = random_gens(N_chains)
    if rand_init == true
        println("Random Initialization")
        chains = [Chain(Int8.(rand(1:21, L)), q, rng[n]) for n in 1:N_chains]
    else
        println("In-sample Initialization")
        chains = [Chain(Int8.(D.msa[:,rand(1:Mtot)]), q, rng[n]) for n in 1:N_chains]
    end
        
        
    M = ModelData(chains, V, N_chains, L, pc, q, H, T = T)
    graf = [Graph(L) for head in 1:H]
    full_graf = Graph(L)
    str = [NumSolVar(TT) for head in 1:H]
    
    #model parameters
    K = TT.(zeros(L,L,H)) 
    h = TT.(log.(pseudocount1(D.f1rs, TT, 1e-8, q)))
    J = TT.(zeros(q, L, q, L))

    #initial thermalization of random sample to local fields with 20 gibbs sweeps
    run_gibbs_sampling!(chains, M.msa, h, J, K, V, L, full_graf, 20, N_chains)
     
    #create model statistics
    update_ModelData!(M, V, L, pc, q, H)
    
    savefile !== nothing && (savef = joinpath(savefile, "logH$(H)learn$(learn_r)reg$(reg)pc$(pc)edges$(n_edges)grad$(grad_iter).txt"))
    savefile !== nothing && (file = open(savef,"a"))
    savefile == nothing && (file = "ciao")
    for iter in 1:N_iter
        #from time to time print info on learning
        if iter % each_step == 0
            print_info(D, M, h, J, K, V, graf, reg, iter, q, L, H, structfile, file, savefile) 
        end       
        
        #compute dlog and activate edges accordingly 
        if n_edges !== 0
            get_dlog!(k, y_k, dL, V, D, M, str, reg, q, L, H)
            activate_edges!(k, K, y_k, dL, graf, full_graf, verbose, avoid_upd, opt_k, n_edges, history, iter, order_list)      
        end
        
        n_act = 2*sum([ne(graf[head]) for head in 1:H])
        #gibbs sampling, update the model statistics and do gradient update
        for it in 1:grad_iter
            run_gibbs_sampling!(chains, M.msa, h, J, K, V, L, full_graf, sweeps, N_chains)
            update_ModelData!(M, V, L, pc, q, H)
            if it%10 == 0 || it == 1
                savefile !== nothing && println(file, "")
                savefile !== nothing && println(file, "Grad_iter $(it) norm_g $(round(sqrt(sum(abs2, D.mheads[K .!= 0.] - M.mheads[K .!= 0,] .- 2 .* reg  .* K[K .!= 0.] ))/n_act, digits = 7)) max_g $(round(maximum(abs.(D.mheads[K .!= 0.] - M.mheads[K .!= 0.] .- 2 .* reg  .* K[K .!= 0.])), digits = 7))") 
            end
            if grad_upd == true
                grad_update!(h, K, D, M, graf, learn_r, reg, H, file, savefile, it, history)                
            end
        end 
    end
    get_J!(J, K, V)
    savefile !== nothing && close(file)
    return (K = K, h = h, J = J, str = str, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains, history = history, reg = reg, pc = pc, order_list = order_list)
end

function grad_update!(h::Array{T,2}, K::Array{T,3}, D, M, graf, learn_r::T, reg::T, H::Int, file, savefile::Union{String, Nothing}, it::Int, history::Array{Int,3}) where {T}
    #gradient descent on fields
    h .+= learn_r .* (D.f1rs .- M.f1rs)
    #gradient descent on couplings only for activated edges
    for head in 1:H
        if ne(graf[head]) !== 0
            for i in findall(.!isempty.(graf[head].fadjlist))
                for j in neighbors(graf[head], i)
                    if (j>i && it%10 == 0) || (j>i && it == 1)
                        savefile !== nothing && println(file, "Edge $i $j $head hist $(history[i,j,head]) K $(K[i,j,head]) Grad $(D.mheads[i,j,head] - M.mheads[i,j,head] - 2 * reg *K[i, j, head])")
                    end
                    K[i, j, head] += learn_r * (D.mheads[i,j,head] - M.mheads[i,j,head] - 2 * reg *K[i, j, head])
                end
            end
        end
    end 
end
                        
                        
function grad_update!(h::Array{T,2}, K::Array{T,3}, D, M, graf, learn_r::T, reg::T, H::Int) where {T}
    #gradient descent on fields
    h .+= learn_r .* (D.f1rs .- M.f1rs)
    #gradient descent on couplings only for activated edges
    for head in 1:H
        if ne(graf[head]) !== 0
            for i in findall(.!isempty.(graf[head].fadjlist))
                for j in neighbors(graf[head], i)
                    K[i, j, head] += learn_r * (D.mheads[i,j,head] - M.mheads[i,j,head] - 2 * reg *K[i, j, head])
                end
            end
        end
    end 
end



function print_info(D, M, h::Array{T,2}, J::Array{T,4}, K::Array{T,3}, V::Array{T,3}, graf, reg::T, iter::Int, q::Int, L::Int, H::Int, structfile::String, file, savefile) where {T}
    get_J!(J, K, V)
    s = score(K,V)
    PPV = compute_PPV(s,structfile)
    n_act = 2*sum([ne(graf[head]) for head in 1:H])
    potts_par = q*q*L*(L-1)/2 + L*q
    println()
    @info "N_Iter $(iter) One $(round(cor(M.f1[:],D.f1[:]), digits = 3)) Conn $(round(cor(triu(M.f2 - M.f1*M.f1', 21)[:], triu(D.f2 - D.f1*D.f1', 21)[:]), digits = 3)) Conn head $(round(cor(D.mheads[:] .- D.mheads_disc[:], M.mheads[:] .- M.mheads_disc[:]), digits = 3)) PPV@L $(round(PPV[L], digits = 3)) PPV@2L $(round(PPV[2*L], digits = 3)) #edges $(sum([ne(graf[head]) for head in 1:H])) / $(Int(L*(L-1)*H/2)) %2PLM = $(round((L*q+ q*H +n_act)/potts_par, digits = 3))" 
    savefile !== nothing && println(file,"")
    savefile !== nothing && println(file, "N_Iter $(iter) One $(round(cor(M.f1[:],D.f1[:]), digits = 3)) Conn $(round(cor(triu(M.f2 - M.f1*M.f1', 21)[:], triu(D.f2 - D.f1*D.f1', 21)[:]), digits = 3)) Conn head $(round(cor(D.mheads[:] .- D.mheads_disc[:], M.mheads[:] .- M.mheads_disc[:]), digits = 3)) PPV@L $(round(PPV[L], digits = 3)) PPV@2L $(round(PPV[2*L], digits = 3)) #edges $(sum([ne(graf[head]) for head in 1:H])) / $(Int(L*(L-1)*H/2)) %2PLM = $(round((L*q+ q*H +n_act)/potts_par, digits = 3))")
    if iter > 1 && sum(K .!= 0.)>0
        println("Iter $(iter) norm_g $(round(sqrt(sum(abs2, D.mheads[K .!= 0.] - M.mheads[K .!= 0,] .- 2 .* reg  .* K[K .!= 0.] ))/n_act, digits = 7)) max_g $(round(maximum(abs.(D.mheads[K .!= 0.] - D.mheads[K .!= 0.] .- 2 .* reg  .* K[K .!= 0.])), digits = 7))") 
    end
end


function runSparseHop(m, V::Array{T,3}; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 1e-2, 
        each_step = 10, q = 21, n_edges = 30, avoid_upd = false, verbose = false, opt_k = true, grad_upd = true) where {T}
    
    
    K = m.K; h = m.h; J = m.J; str = m.str; graf = m.graf; 
    full_graf = m.full_graf; D = m.D; M = m.M; chains = m.chains; N_chains = length(chains); history = m.history;
    pc = m.pc; reg = m.reg;
    
    #global variables
    TT = eltype(V); pc = TT(pc); reg = TT(reg); learn_r = TT(learn_r); 
    H = size(V,3)
    L = size(D.msa,1) 
    
    #containers
    dL = TT.(zeros(L,L,H)); k = TT.(zeros(L,L,H)); y_k = TT.(zeros(L,L,H));
   
    #initial thermalization of random sample to local fields with 20 gibbs sweeps
    run_gibbs_sampling!(chains, M.msa, h, J, K, V, L, full_graf, 20, N_chains)
     
    #create model statistics
    update_ModelData!(M, V, L, pc, q, H)
    
    for iter in 1:N_iter
        #from time to time print info on learning
        if iter % each_step == 0
            print_info(D, M, h, J, K, V, graf, reg, iter, q, L, H, structfile) 
        end       
        
        #compute dlog and activate edges accordingly 
        if n_edges !== 0
            get_dlog!(k, y_k, dL, V, D, M, str, reg, q, L, H)
            activate_edges!(k, K, y_k, dL, graf, full_graf, verbose, avoid_upd, opt_k, n_edges)      
        end
        
        #gibbs sampling, update the model statistics and do gradient update
        for it in 1:grad_iter
            run_gibbs_sampling!(chains, M.msa, h, J, K, V, L, full_graf, sweeps, N_chains)
            update_ModelData!(M, V, L, pc, q, H)
            if grad_upd == true
                grad_update!(h, K, D, M, graf, learn_r, reg, H)                
            end 
        end 
    end
    get_J!(J, K, V)
    return (K = K, h = h, J = J, str = str, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains)
end



