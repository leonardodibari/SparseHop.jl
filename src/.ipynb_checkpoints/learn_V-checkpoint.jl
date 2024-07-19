function new_runSparseHop_onV(out, V; 
        msa_file = "../DataAttentionDCA/data/PF00014/PF00014_mgap6.fasta.gz", 
        structfile = "../DataAttentionDCA/data/PF00014/PF00014_struct.dat", 
        N_iter::Int = 100, sweeps::Int = 5, learn_r = 1e-2, 
        each_step = 10, q = 21, savefile::Union{String, Nothing} = nothing, savepars = true) 
    
    #global variables
    K = deepcopy(out.K); h = deepcopy(out.h);  J = deepcopy(out.J); graf = deepcopy(out.graf); full_graf = deepcopy(out.full_graf); D = deepcopy(out.D); M = deepcopy(out.M); chains = deepcopy(out.chains); reg = deepcopy(out.reg); pc = deepcopy(out.pc);
    #V = deepcopy(out.V);
    
    TT = eltype(V); learn_r = TT(learn_r); 
    H = size(V,3)
    L,Mtot = size(D.msa) 
    N_chains = length(chains);
    
    pars_file = "../parsH$(H)learn$(learn_r)reg$(reg)pc$(pc).jld2"
    
    savefile !== nothing && (savef = joinpath(savefile, "logH$(H)learn$(learn_r)reg$(reg)pc$(pc).txt"))
    savefile !== nothing && (file = open(savef,"a"))
    savefile == nothing && (file = "ciao")
    for iter in 1:N_iter
            
        #from time to time print info on learning
        if iter % each_step == 0
            print_info(D, M, h, J, K, V, graf, reg, iter, q, L, H, structfile, file, savefile)
        end       
        
        #gibbs sampling, update the model statistics and do gradient updateon V
        run_gibbs_sampling!(chains, M.msa, h, J, K, V, L, full_graf, sweeps, N_chains)
        update_ModelData!(M, V, L, pc, q, H)
        grad_update_onV!(h, K, V, D, M, graf, learn_r, reg, H)                
                
        if savepars == true 
            out = (K = K, h = h, J = J, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains, reg = reg, pc = pc)
            @save pars_file out
        end
    end
    get_J!(J, K, V)
    savefile !== nothing && close(file)
    return (K = K, h = h, V = V, J = J, graf = graf, full_graf = full_graf, D = D, M = M, chains = chains, reg = reg, pc = pc)
end

function grad_update_onV!(h::Array{T,2}, K::Array{T,3}, V::Array{T,3}, D, M, graf, learn_r::T, reg::T, H::Int) where {T}
    #gradient descent on fields
    h .+= learn_r .* (D.f1rs .- M.f1rs)
    #gradient descent on couplings only for activated edges
    for head in 1:H
        if ne(graf[head]) !== 0
            for a in 1:21
                for b in 1:21
                    sumD = zero(T); sumM = zero(T);
                    for i in findall(.!isempty.(graf[head].fadjlist))
                        for j in neighbors(graf[head], i)
                            sumD += K[i,j,head]*D.f2rs[a,i,b,j]
                            sumM += K[i,j,head]*M.f2rs[a,i,b,j]
                        end
                    end
                    V[a, b, head] += learn_r * (sumD - sumM)
                end
            end
        end
    end 
end
