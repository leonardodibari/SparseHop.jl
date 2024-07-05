
function new_runSparseHop(V::Array{T,3}; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_chains = 5000, N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 1e-2, 
        each_step = 10, q = 21, pc = 1e-2, n_edges = 30, reg = 1e-2, rand_init = true, avoid_upd = false, verbose = false, opt_k = true, grad_upd = true, savefile::Union{String, Nothing} = nothing, savepars = true) where {T}
    
    #global variables
    TT = eltype(V); pc = TT(pc); reg = TT(reg); learn_r = TT(learn_r); 
    H = size(V,3)
    D = Data(msa_file, V, pc, q, H, T = T)
    L,Mtot = size(D.msa) 
    
    #containers
    dL = TT.(zeros(L,L,H)); k = TT.(zeros(L,L,H)); y_k = TT.(zeros(L,L,H)); history = Int.(zeros(L,L,H)); 
    order_list = [];
    new_type =Optim.MultivariateOptimizationResults{Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.HagerZhang{Float64, Base.RefValue{Bool}},Optim.var"#19#21"},ComponentArrays.ComponentArray{T,1,Array{T,1},Tuple{ComponentArrays.Axis{(hm = 1:21, hn = 22:42, Kmn = 43)}}},T,T,Array{Optim.OptimizationState{Float32,Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.HagerZhang{Float64, Base.RefValue{Bool}},Optim.var"#19#21"}},1},Bool,NamedTuple{(:f_limit_reached, :g_limit_reached, :h_limit_reached, :time_limit, :callback, :f_increased),Tuple{Bool,Bool,Bool,Bool,Bool,Bool}}}
    
    #=Optim.MultivariateOptimizationResults{Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.BackTracking{Float64, Int64},Optim.var"#20#22"},ComponentArrays.ComponentArray{T,1,Array{T,1},Tuple{ComponentArrays.Axis{(hm = 1:21, hn = 22:42, Kmn = 43)}}},T,T,Array{Optim.OptimizationState{T,Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.BackTracking{Float64, Int64},Optim.var"#20#22"}},1},Bool,NamedTuple{(:f_limit_reached, :g_limit_reached, :h_limit_reached, :time_limit, :callback, :f_increased),Tuple{Bool,Bool,Bool,Bool,Bool,Bool}}}=#
    
    
    
    minim_res = Array{new_type}(undef, (L,L,H));
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
    graf = [Graph(L) for head in 1:H]
    full_graf = Graph(L)
    
    #model parameters
    K = TT.(zeros(L,L,H)) 
    h = TT.(log.(pseudocount1(D.f1rs, TT, 1e-8, q)))
    J = TT.(zeros(q, L, q, L))

    #initial thermalization of random sample to local fields with 20 gibbs sweeps
    run_gibbs_sampling!(chains, M.msa, h, J, K, V, L, full_graf, 20, N_chains)
     
    #create model statistics
    update_ModelData!(M, V, L, pc, q, H)
    
    pars_file = "parsN_iter$(N_iter)H$(H)learn$(learn_r)reg$(reg)pc$(pc)edges$(n_edges)grad$(grad_iter).jld2"
    
    savefile !== nothing && (savef = joinpath(savefile, "logH$(H)learn$(learn_r)reg$(reg)pc$(pc)edges$(n_edges)grad$(grad_iter).txt"))
    savefile !== nothing && (file = open(savef,"a"))
    savefile == nothing && (file = "ciao")
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
    return (K = K, h = h, J = J, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains, history = history, reg = reg, pc = pc, order_list = order_list)
end

function new_grad_update!(h::Array{T,2}, K::Array{T,3}, D, M, graf, learn_r::T, reg::T, H::Int, file, savefile::Union{String, Nothing}, it::Int, history::Array{Int,3}) where {T}
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
                        
                        


function new_runSparseHop(out, V::Array{T,3}; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_iter::Int = 100, grad_iter::Int = 1, sweeps::Int = 5, learn_r = 1e-2, 
        each_step = 10, q = 21, n_edges = 30, avoid_upd = false, verbose = false, opt_k = true, grad_upd = true, savefile::Union{String, Nothing} = nothing, savepars = true) where {T}
    
    #global variables
    K = deepcopy(out.K); h = deepcopy(out.h); J = deepcopy(out.J); graf = deepcopy(out.graf); full_graf = deepcopy(out.full_graf); D = deepcopy(out.D); M = deepcopy(out.M); chains = deepcopy(out.chains); reg = deepcopy(out.reg); pc = deepcopy(out.pc)
    
    TT = eltype(V); learn_r = TT(learn_r); 
    H = size(V,3)
    L,Mtot = size(D.msa) 
    N_chains = length(chains);
    
    #containers
    dL = TT.(zeros(L,L,H)); k = TT.(zeros(L,L,H)); y_k = TT.(zeros(L,L,H)); history = Int.(zeros(L,L,H)); 
    order_list = [];
    new_type =Optim.MultivariateOptimizationResults{Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.HagerZhang{Float64, Base.RefValue{Bool}},Optim.var"#19#21"},ComponentArrays.ComponentArray{T,1,Array{T,1},Tuple{ComponentArrays.Axis{(hm = 1:21, hn = 22:42, Kmn = 43)}}},T,T,Array{Optim.OptimizationState{Float32,Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.HagerZhang{Float64, Base.RefValue{Bool}},Optim.var"#19#21"}},1},Bool,NamedTuple{(:f_limit_reached, :g_limit_reached, :h_limit_reached, :time_limit, :callback, :f_increased),Tuple{Bool,Bool,Bool,Bool,Bool,Bool}}}
    
    #=Optim.MultivariateOptimizationResults{Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.BackTracking{Float64, Int64},Optim.var"#20#22"},ComponentArrays.ComponentArray{T,1,Array{T,1},Tuple{ComponentArrays.Axis{(hm = 1:21, hn = 22:42, Kmn = 43)}}},T,T,Array{Optim.OptimizationState{T,Optim.LBFGS{Nothing,LineSearches.InitialStatic{Float64},LineSearches.BackTracking{Float64, Int64},Optim.var"#20#22"}},1},Bool,NamedTuple{(:f_limit_reached, :g_limit_reached, :h_limit_reached, :time_limit, :callback, :f_increased),Tuple{Bool,Bool,Bool,Bool,Bool,Bool}}}=#
    
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
                    ps[i,j,head] = ConstPara(V[:, :, head], D.f2rs[:, i, :, j], M.f2rs[:, i, :, j], T(2*reg))
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
    end
    get_J!(J, K, V)
    savefile !== nothing && close(file)
    return (K = K, h = h, J = J, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains, history = history, reg = reg, pc = pc, order_list = order_list)
end
