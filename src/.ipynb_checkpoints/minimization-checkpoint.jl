

struct ConstPara{T}
    enu::Matrix{T}
    Δmn::Matrix{T}
    Ptilde::Matrix{T}
    fmn::Matrix{T}
    Pmn::Matrix{T}
    fm::Vector{T}
    fn::Vector{T}
    lambda::T
end


function ConstPara(enu::Matrix{T}, fmn::Matrix{T}, Pmn::Matrix{T}, lambda::T) where T<: AbstractFloat
    q = 21
    fm = sum(fmn, dims=2)[:]
    fn = sum(fmn, dims=1)[:]
    @assert sum(fm)≈1
    @assert sum(fn)≈1
    Δmn = zeros(T, q, q)
    Ptilde = zeros(T, q, q)
    return ConstPara{T}(enu, Δmn, Ptilde, fmn, Pmn, fm, fn, lambda)
end

function read_param(njld2::String; m::Int=17, n::Int=41, nu::Int=15, lambda::Real=0.2)
    jld2 = JLD2.load(njld2)
    enu = jld2["V"]
    fij = jld2["f_emp"]
    Pij = jld2["f_mod"]
    T = eltype(enu)
    return ConstPara(enu[:, :, nu], fij[:, m, :, n], Pij[:, m, :, n],T(lambda))
end

function fg!(F, G, x, p::ConstPara{T}) where T<: AbstractFloat
    maxval = typemin(T)
    ll1 = T(0)
    @inbounds @simd for b in eachindex(x.hn)
        for a in eachindex(x.hm)
            p.Δmn[a,b] = x.hm[a] + x.hn[b] + x.Kmn * p.enu[a, b]
            ll1 += p.Δmn[a, b] * p.fmn[a, b]
            maxval = ifelse(p.Δmn[a, b] > maxval, p.Δmn[a, b], maxval)
        end
    end
    nrm = T(0)
    @inbounds @simd for b in eachindex(x.hn)
        for a in eachindex(x.hm)
            p.Ptilde[a, b] = exp(p.Δmn[a, b]-maxval) * p.Pmn[a, b]
            nrm += p.Ptilde[a, b]
        end
    end
    p.Ptilde ./= nrm
    if G !== nothing 
        llptilde = T(0)
        llenu = T(0)
        @inbounds @simd for c in eachindex(G.hn, G.hm)
            G.hm[c] = p.fm[c] - x.hm[c]*p.lambda
            G.hn[c] = p.fn[c] - x.hn[c]*p.lambda
            for a in eachindex(G.hm)
                llenu += p.fmn[a, c] * p.enu[a, c]
                llptilde += p.Ptilde[a, c] * p.enu[a, c]
                G.hm[c] -= p.Ptilde[c, a] 
                G.hn[c] -= p.Ptilde[a, c]
            end
        end
        G.Kmn = llenu - llptilde - p.lambda*x.Kmn
        G .*=-1
    end
    ## Compute the likelihood and return it
    if F !== nothing
        return -(ll1 - maxval - log(nrm) - p.lambda * sum(abs2,x)/2)
    end
end


function optimize(p::ConstPara{T}; 
    x0=ComponentArray{T}(hm=zeros(T,21),hn=zeros(T,21),Kmn=zero(T)),
    show_trace=true, 
    iterations=1000, 
    tol=1e-6,
    optimizer=LBFGS()) where T<: AbstractFloat

    wrapfg! = (F, G, x) -> fg!(F , G, x, p)
    res = Optim.optimize(Optim.only_fg!(wrapfg!), 
        x0,
        optimizer, 
        Optim.Options(show_trace=show_trace, iterations=iterations, f_tol=tol))
    return res
end


function new_get_dlog!(ps, minim_res, k::Array{T,3}, dL::Array{T,3}, L::Int, H::Int) where {T}
    
    emme = 0
    enne = 0 
    nu = 0
    @tasks for i in 1:L
        for j in (i+1):L
            for head in 1:H
                try 
                    minim_res[i,j,head] = SparseHop.optimize(ps[i,j,head], show_trace = false)
                catch e
                    emme = deepcopy(i); enne = deepcopy(j); nu = deepcopy(head);
                    err = (ps = ps, minim_res = minim_res, idxs = [emme, enne, nu])
                    #println(ps[emme,enne,nu])
                    println(minim_res[emme,enne,nu])                    
                    @save "minim_error.jld2" err
                    error("Error occured at (m,n,nu) = ($(emme), $(enne), $(nu))")
                end
                dL[i,j,head] = -minim_res[i,j,head].minimum
                k[i,j,head] = minim_res[i,j,head].minimizer[end]
            end
        end
    end  
end

function debug_new_get_dlog!(ps, minim_res, k::Array{T,3}, dL::Array{T,3}, L::Int, H::Int) where {T}
    
    emme = 0
    enne = 0 
    nu = 0
    @tasks for i in 1:L
        for j in (i+1):L
            for head in 1:H
                emme = deepcopy(i); enne = deepcopy(j); nu = deepcopy(head);
                try 
                    if j == 10
                        minim_res[i,j,head] = SparseHop.optimize(pss[i,j,head], show_trace = false)
                    else 
                        minim_res[i,j,head] = SparseHop.optimize(ps[i,j,head], show_trace = false)
                    end
                catch e
                    err = (ps = ps, minim_res = minim_res, idxs = [emme, enne, nu])
                    error("Error occured at (m,n,nu) = ($(emme), $(enne), $(nu))")
                    @save "minim_error.jld2" err
                end
                dL[i,j,head] = -minim_res[i,j,head].minimum
                k[i,j,head] = minim_res[i,j,head].minimizer[end]
            end
        end
    end  
end


function new_activate_edges!(k::Array{T,3}, K::Array{T,3}, h::Array{T,2}, minim_res, dL::Array{T,3}, graf, full_graf, verbose, avoid_upd, opt_k, n_edges::Int, history::Array{Int,3}, iter, order_list) where {T}
    
    #=L, L, H = size(K)
    all_edges = round(Int,L*(L-1)*H/2)
    
    copy_dL = deepcopy(dL)
    order = []
    for _ in 1:all_edges
        m, n, nu = Tuple(argmax(copy_dL))
        push!(order, Tuple(argmax(copy_dL)))
        copy_dL[m,n,nu] = 0
    end   
    push!(order_list, order)
    =#
    if avoid_upd == true
        dL[K .!= 0.] .= T(0.)
    end
            
    for _ in 1:n_edges
        m, n, nu = Tuple(argmax(dL))
        history[m,n,nu] = iter
        if verbose == true
            println(" Suggested K : $(k[m,n,nu]), terms $(argmax(dL)), dL : $(maximum(dL))")
        end
        if opt_k == true
            K[m,n,nu] += k[m,n,nu]
            for a in 1:21
                h[a,m] += minim_res[m,n,nu].minimizer[a] 
                h[a,n] += minim_res[m,n,nu].minimizer[21+a]
            end
            K[n,m,nu] += K[m,n,nu]
        end
        add_edge!(graf[nu], m, n)
        add_edge!(full_graf, m, n)
        dL[m,n,nu] = T(0.)
    end
    dL .= T(0.)
end
    
       
# from here on is the simple version of the optimization using Zygote (much
# slower). We get the same results as the previous version.

function f(x, p::ConstPara{T}) where {T<:AbstractFloat}
    ll1 = -p.lambda * sum(abs2, x) / 2
    @inbounds @simd for b in eachindex(x.hn)
        for a in eachindex(x.hm)
            ll1 += (x.hm[a] + x.hn[b] + x.Kmn * p.enu[a, b]) * p.fmn[a, b]
        end
    end
    nrm = T(0)
    @inbounds @simd for b in eachindex(x.hn)
        for a in eachindex(x.hm)
            nrm += exp(x.hm[a] + x.hn[b] + x.Kmn * p.enu[a, b]) * p.Pmn[a, b]
        end
    end
    #println("likelihood = $(ll1 - log(nrm)) Km = $(x.Kmn)")
    return (ll1 - log(nrm))
end

function g!(G, x, p)
    G .= Zygote.gradient(x -> f(x, p), x)[1]
end

function optimize_simple_value(p::ConstPara{T};
    x0=ComponentArray{T}(hm=randn(T, 21), hn=randn(T, 21), Kmn=randn(T)),
    show_trace=true,
    iterations=1000,
    tol=1e-6,
    optimizer=LBFGS(linesearch = LineSearches.BackTracking())) where {T<:AbstractFloat}
    # initial_Kmn = x0.Kmn

    wrapf = x -> f(x, p)
    wrapg! = (G,x) -> g!(G,x,p)

    res = Optim.maximize(wrapf, wrapg!, x0, optimizer, Optim.Options(show_trace=show_trace, iterations=iterations, f_tol=tol))
    return res
    # return res, x0
end









