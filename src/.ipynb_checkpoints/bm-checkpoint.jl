
function new_run_bm(V::Array{T,3}; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_chains = 5000, N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 1e-2, 
        each_step = 10, q = 21, pc = 1e-2, reg = 1e-2, rand_init = true, savefile::Union{String, Nothing} = nothing, savepars = false) where {T}
    
    #global variables
    TT = eltype(V); pc = TT(pc); reg = TT(reg); learn_r = TT(learn_r); 
    H = size(V,3)
    D = Data(msa_file, V, pc, q, H, T = T)
    L,Mtot = size(D.msa) 
    
    potts_par = q*q*L*(L-1)/2 + L*q
     
    ps = Array{ConstPara}(undef, (L,L,H));
    
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
    
    
    pars_file = "parsN_iter$(N_iter)H$(H)learn$(learn_r)reg$(reg)pc$(pc)grad$(grad_iter).jld2"
    
    savefile !== nothing && (savef = joinpath(savefile, "logH$(H)learn$(learn_r)reg$(reg)pc$(pc)grad$(grad_iter).txt"))
    savefile !== nothing && (file = open(savef,"a"))
    savefile == nothing && (file = "ciao")
    
    #model parameters
    K = TT.(zeros(L,L,H)) 
    h = TT.(log.(pseudocount1(D.f1rs, TT, 1e-8, q)))
    J = TT.(zeros(q, L, q, L))
    
    update_ModelData!(M, V, L, pc, q, H)
    
    print_info_bm(D, M, h, J, K, V, reg, 0, q, L, H, structfile, file, savefile)

    #initial thermalization of random sample to local fields with 20 gibbs sweeps
    
    run_gibbs_sampling_bm!(chains, M.msa, h, J, K, V, L, 20, N_chains)
     
    #create model statistics
    update_ModelData!(M, V, L, pc, q, H)
    
    #=pars_file = "parsN_iter$(N_iter)H$(H)learn$(learn_r)reg$(reg)pc$(pc)grad$(grad_iter).jld2"
    
    savefile !== nothing && (savef = joinpath(savefile, "logH$(H)learn$(learn_r)reg$(reg)pc$(pc)grad$(grad_iter).txt"))
    savefile !== nothing && (file = open(savef,"a"))
    savefile == nothing && (file = "ciao")=#
    
    for iter in 1:N_iter
        for i in 1:L
            for j in (i+1):L
                for head in 1:H
                    ps[i,j,head] = ConstPara(V[:, :, head], D.f2rs[:, i, :, j], M.f2rs[:, i, :, j], T(2*reg))
                end
            end
        end
    
        #from time to time print info on learning
        if iter % each_step == 0
            print_info_bm(D, M, h, J, K, V, reg, iter, q, L, H, structfile, file, savefile)
        end       
        
               
        #gibbs sampling, update the model statistics and do gradient update
        for it in 1:grad_iter
        
            run_gibbs_sampling_bm!(chains, M.msa, h, J, K, V, L, sweeps, N_chains)
            update_ModelData!(M, V, L, pc, q, H)
            if it%10 == 0 || it == 1
                savefile !== nothing && println(file, "")
                savefile !== nothing && println(file, "Grad_iter $(it) norm_g $(round(sqrt(sum(abs2, D.mheads - M.mheads .- 2 .* reg  .* K ))/potts_par, digits = 7)) max_g $(round(maximum(abs.(D.mheads - M.mheads .- 2 .* reg  .* K)), digits = 7))") 
            end
            
            ## CHANGE THIS
            grad_update_bm!(h, K, D, M, learn_r, reg, H, L, file, savefile)                
        end 
        if savepars == true 
            out = (K = K, h = h, J = J, D = D, M = M, chains = chains, reg = reg, pc = pc)
            @save pars_file out
        end
    end
    get_J!(J, K, V)
    savefile !== nothing && close(file)
    return (K = K, h = h, V = V, J = J, D = D, M = M, chains = chains, reg = reg, pc = pc)
end


function grad_update_bm!(h::Array{T,2}, K::Array{T,3}, D, M, learn_r::T, reg::T, H::Int, L::Int, file, savefile::Union{String, Nothing}) where {T}
    #gradient descent on fields
    h .+= learn_r .* (D.f1rs .- M.f1rs)
    #gradient descent on couplings only for activated edges
    for head in 1:H
        for i in 1:L
            for j in i+1:L
                K[i, j, head] += learn_r * (D.mheads[i,j,head] - M.mheads[i,j,head] - 2 * reg *K[i, j, head])   
            end
        end
    end 
end
                        
   

function prob_cond_bm!(chain, 
        site::Int, 
        h::Array{T,2}, 
        J::Array{T,4}, 
        L::Int) where {T}
    
    fill!(chain.log_prob, T(0))
	@inbounds for a in 1:21
		chain.log_prob[a] += h[a, site]
 		for j in 1:L
			chain.log_prob[a] += J[chain.seq[j], j, a, site]
        end
	end
    loc_softmax!(chain.log_prob)
end



function gibbs_sampling_bm!(chain, h::Array{T,2}, J::Array{T,4}, L::Int, sweeps::Int) where {T}
    @inbounds for s in 1:sweeps
        for site in shuffle!(chain.sites)
            prob_cond_bm!(chain, site, h, J, L)
            loc_sample!(chain.generator, chain.log_prob, chain.seq, site)
        end
    end
end


function run_gibbs_sampling_bm!(chains, msa::Array{Int8,2}, h::Array{T,2}, J::Array{T,4}, K::Array{T,3}, V::Array{T,3},L::Int, sweeps::Int, N_chains::Int) where {T}
    
    get_J!(J, K, V)
    @tasks for n in 1:N_chains
        gibbs_sampling_bm!(chains[n], h, J, L, sweeps)
        for i in 1:L
           msa[i,n] = chains[n].seq[i]
        end
    end
    
end




      


function print_info_bm(D, M, h::Array{T,2}, J::Array{T,4}, K::Array{T,3}, V::Array{T,3}, reg::T, iter::Int, q::Int, L::Int, H::Int, structfile::String, file, savefile) where {T}
    get_J!(J, K, V)
    s = score(K,V)
    PPV = compute_PPV(s,structfile)
    @tullio alt_score[i,j] := K[i,j,head]*K[i,j,head]
    s_new = SparseHop.compute_ranking(alt_score);
    PPV_new = compute_PPV(s_new,structfile)
    potts_par = q*q*L*(L-1)/2 + L*q
    println()
    @info "N_Iter $(iter) One $(round(cor(M.f1[:],D.f1[:]), digits = 3)) Conn $(round(cor(triu(M.f2 - M.f1*M.f1', 21)[:], triu(D.f2 - D.f1*D.f1', 21)[:]), digits = 3)) Conn head $(round(cor(D.mheads[:] .- D.mheads_disc[:], M.mheads[:] .- M.mheads_disc[:]), digits = 3)) newPPV@L $(round(PPV_new[L], digits = 3)) PPV@L $(round(PPV[L], digits = 3)) PPV@2L $(round(PPV[2*L], digits = 3))" 
    savefile !== nothing && println(file,"")
    savefile !== nothing && println(file, "N_Iter $(iter) One $(round(cor(M.f1[:],D.f1[:]), digits = 3)) Conn $(round(cor(triu(M.f2 - M.f1*M.f1', 21)[:], triu(D.f2 - D.f1*D.f1', 21)[:]), digits = 3)) Conn head $(round(cor(D.mheads[:] .- D.mheads_disc[:], M.mheads[:] .- M.mheads_disc[:]), digits = 3)) PPV@L $(round(PPV[L], digits = 3)) PPV@2L $(round(PPV[2*L], digits = 3)) ")
    if iter > 1 && sum(K .!= 0.)>0
        println("Iter $(iter) norm_g $(round(sqrt(sum(abs2, D.mheads - M.mheads .- 2 .* reg  .* K ))/potts_par, digits = 7)) max_g $(round(maximum(abs.(D.mheads - D.mheads .- 2 .* reg  .* K)), digits = 7))") 
    end
end






function new_runSparseHop_bm(out; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 1e-2, 
        each_step = 10, q = 21, n_edges = 30, avoid_upd = false, verbose = false, opt_k = true, grad_upd = true, savefile::Union{String, Nothing} = nothing, savepars = false) 
    
    #global variables
    K = deepcopy(out.K); h = deepcopy(out.h); V = deepcopy(out.V); J = deepcopy(out.J); graf = deepcopy(out.graf); full_graf = deepcopy(out.full_graf); D = deepcopy(out.D); M = deepcopy(out.M); chains = deepcopy(out.chains); reg = deepcopy(out.reg); pc = deepcopy(out.pc)
    
    TT = eltype(V); learn_r = TT(learn_r); 
    H = size(V,3)
    L,Mtot = size(D.msa) 
    N_chains = length(chains);
    
    #containers
    dL = TT.(zeros(L,L,H)); k = TT.(zeros(L,L,H)); y_k = TT.(zeros(L,L,H)); history = Int.(zeros(L,L,H)); 
    order_list = [];
    new_type =Optim.MultivariateOptimizationResults{Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.HagerZhang{Float64, Base.RefValue{Bool}},Optim.var"#19#21"},ComponentArrays.ComponentArray{TT,1,Array{TT,1},Tuple{ComponentArrays.Axis{(hm = 1:21, hn = 22:42, Kmn = 43)}}},TT,TT,Array{Optim.OptimizationState{TT,Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.HagerZhang{Float64, Base.RefValue{Bool}},Optim.var"#19#21"}},1},Bool,NamedTuple{(:f_limit_reached, :g_limit_reached, :h_limit_reached, :time_limit, :callback, :f_increased),Tuple{Bool,Bool,Bool,Bool,Bool,Bool}}}
    
    minim_res = Array{new_type}(undef, (L,L,H));
    ps = Array{ConstPara}(undef, (L,L,H));
    
    pars_file = "../parsH$(H)learn$(learn_r)reg$(reg)pc$(pc)edges$(n_edges)grad$(grad_iter).jld2"
    
    savefile !== nothing && (savef = joinpath(savefile, "logH$(H)learn$(learn_r)reg$(reg)pc$(pc)edges$(n_edges)grad$(grad_iter).txt"))
    savefile !== nothing && (file = open(savef,"a"))
    savefile == nothing && (file = "ciao")
    for iter in 1:N_iter
        for i in 1:L
            for j in (i+1):L
                for head in 1:H
                    ps[i,j,head] = ConstPara(V[:, :, head], D.f2rs[:, i, :, j], M.f2rs[:, i, :, j], TT(2*reg))
                end
            end
        end
    
        #from time to time print info on learning
        if iter % each_step == 0
            print_info(D, M, h, J, K, V, graf, reg, iter, q, L, H, structfile, file, savefile)
        end       
        
        #compute dlog and activate edges accordingly 
        if n_edges !== 0
            new_get_dlog!(ps, minim_res, k, dL, L, H)
            new_activate_edges!(k, K, h, minim_res, dL, graf, full_graf, verbose, avoid_upd, opt_k, n_edges, history, iter, order_list)      
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
        if savepars == true 
            out = (K = K, h = h, J = J, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains, reg = reg, pc = pc)
            @save pars_file out
        end
    end
    get_J!(J, K, V)
    savefile !== nothing && close(file)
    return (K = K, h = h, V = V, J = J, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains, history = history, reg = reg, pc = pc, order_list = order_list)
end


function new_runSparseHop_nostruct_bm(out; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 1e-2, 
        each_step = 10, q = 21, n_edges = 30, avoid_upd = false, verbose = false, opt_k = true, grad_upd = true, savefile::Union{String, Nothing} = nothing, savepars = false) 
    
    #global variables
    K = deepcopy(out.K); h = deepcopy(out.h); V = deepcopy(out.V); J = deepcopy(out.J); graf = deepcopy(out.graf); full_graf = deepcopy(out.full_graf); D = deepcopy(out.D); M = deepcopy(out.M); chains = deepcopy(out.chains); reg = deepcopy(out.reg); pc = deepcopy(out.pc)
    
    TT = eltype(V); learn_r = TT(learn_r); 
    H = size(V,3)
    L,Mtot = size(D.msa) 
    N_chains = length(chains);
    
    #containers
    dL = TT.(zeros(L,L,H)); k = TT.(zeros(L,L,H)); y_k = TT.(zeros(L,L,H)); history = Int.(zeros(L,L,H)); 
    order_list = [];
    new_type =Optim.MultivariateOptimizationResults{Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.HagerZhang{Float64, Base.RefValue{Bool}},Optim.var"#19#21"},ComponentArrays.ComponentArray{TT,1,Array{TT,1},Tuple{ComponentArrays.Axis{(hm = 1:21, hn = 22:42, Kmn = 43)}}},TT,TT,Array{Optim.OptimizationState{TT,Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.HagerZhang{Float64, Base.RefValue{Bool}},Optim.var"#19#21"}},1},Bool,NamedTuple{(:f_limit_reached, :g_limit_reached, :h_limit_reached, :time_limit, :callback, :f_increased),Tuple{Bool,Bool,Bool,Bool,Bool,Bool}}}
    
    minim_res = Array{new_type}(undef, (L,L,H));
    ps = Array{ConstPara}(undef, (L,L,H));
    
    pars_file = "../parsH$(H)learn$(learn_r)reg$(reg)pc$(pc)edges$(n_edges)grad$(grad_iter).jld2"
    
    savefile !== nothing && (savef = joinpath(savefile, "logH$(H)learn$(learn_r)reg$(reg)pc$(pc)edges$(n_edges)grad$(grad_iter).txt"))
    savefile !== nothing && (file = open(savef,"a"))
    savefile == nothing && (file = "ciao")
    for iter in 1:N_iter
        for i in 1:L
            for j in (i+1):L
                for head in 1:H
                    ps[i,j,head] = ConstPara(V[:, :, head], D.f2rs[:, i, :, j], M.f2rs[:, i, :, j], TT(2*reg))
                end
            end
        end
    
        #from time to time print info on learning
        if iter % each_step == 0
            print_info_nostruct(D, M, h, J, K, V, graf, reg, iter, q, L, H, file, savefile)
        end       
        
        #compute dlog and activate edges accordingly 
        if n_edges !== 0
            new_get_dlog!(ps, minim_res, k, dL, L, H)
            new_activate_edges!(k, K, h, minim_res, dL, graf, full_graf, verbose, avoid_upd, opt_k, n_edges, history, iter, order_list)      
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
        if savepars == true 
            out = (K = K, h = h, J = J, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains, reg = reg, pc = pc)
            @save pars_file out
        end
    end
    get_J!(J, K, V)
    savefile !== nothing && close(file)
    return (K = K, h = h, V = V, J = J, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains, history = history, reg = reg, pc = pc, order_list = order_list)
end
