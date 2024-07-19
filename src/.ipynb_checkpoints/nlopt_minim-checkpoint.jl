const scrax = ComponentArray(hm=randn(21),hn=randn(21),Kmn=randn())
const ax = getaxes(scrax)

function fg_nlopt!(x::ComponentArray, G::ComponentArray, p::ConstPara{T}) where {T<:AbstractFloat}
    maxval = typemin(T)
    ll1 = T(0)
    @inbounds @simd for b in eachindex(x.hn)
        for a in eachindex(x.hm)
            p.Δmn[a, b] = x.hm[a] + x.hn[b] + x.Kmn * p.enu[a, b]
            ll1 += p.Δmn[a, b] * p.fmn[a, b]
            maxval = ifelse(p.Δmn[a, b] > maxval, p.Δmn[a, b], maxval)
        end
    end
    nrm = T(0)
    @inbounds @simd for b in eachindex(x.hn)
        for a in eachindex(x.hm)
            p.Ptilde[a, b] = exp(p.Δmn[a, b] - maxval) * p.Pmn[a, b]
            nrm += p.Ptilde[a, b]
        end
    end
    p.Ptilde ./= nrm

    llptilde = T(0)
    llenu = T(0)
    @inbounds @simd for c in eachindex(G.hn, G.hm)
        G.hm[c] = p.fm[c] - x.hm[c] * p.lambda
        G.hn[c] = p.fn[c] - x.hn[c] * p.lambda
        for a in eachindex(G.hm)
            llenu += p.fmn[a, c] * p.enu[a, c]
            llptilde += p.Ptilde[a, c] * p.enu[a, c]
            G.hm[c] -= p.Ptilde[c, a]
            G.hn[c] -= p.Ptilde[a, c]
        end
    end
    G.Kmn = llenu - llptilde - p.lambda * x.Kmn
    G .*= -1
    return -(ll1 - maxval - log(nrm) - p.lambda * sum(abs2, x) / 2)
end

fg_nlopt!(x, g, p) = fg_nlopt!(ComponentArray(x, ax), ComponentArray(g, ax), p)

function optimize_nlopt(p::ConstPara{T}; 
    x0=ComponentArray{T}(hm=randn(T,21),hn=randn(T,21),Kmn=randn(T)),
    show_trace=true,
    tol=T(1e-6)) where T <: AbstractFloat

    ax = getaxes(x0)
    wrapfg! = (x, g) -> fg_nlopt!(x, g, p)
    opt = NLopt.Opt(:LD_LBFGS, 43)
    NLopt.xtol_rel!(opt, tol)
    NLopt.maxeval!(opt, 1000)
    #seed!(1234)
    min_objective!(opt, wrapfg!)
    elapstime = @elapsed (minf, minx, ret) = NLopt.optimize(opt, x0)
    return opt, (elapstime, minf, minx)
end













#=
function new_fg!(x::ComponentArray{T}, G, p::ConstPara{T}) where T<: AbstractFloat
    println("start")
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
    println(-(ll1 - maxval - log(nrm) - p.lambda * sum(abs2,x)/2))
    return -(ll1 - maxval - log(nrm) - p.lambda * sum(abs2,x)/2)
end



function optimize_nlopt(p::ConstPara{T}; 
    x0=ComponentArray{T}(hm=zeros(T,21),hn=zeros(T,21),Kmn=zero(T)),
    iterations=100, 
    tol=1e-6,
    optimizer=:LD_LBFGS) where T<: AbstractFloat

    opt = Opt(optimizer, length(x0))      
    xtol_rel!(opt, tol)
    maxeval!(opt, iterations)
    ax = getaxes(x0)
    x_in = rand(43)
    new_fg2!(x, G, p) = new_fg!(ComponentArray(x, ax), ComponentArray(G, ax), p)
    wrap_fg! = (x, G) -> new_fg2!(x, G, p)
    min_objective!(opt, wrap_fg!)  
    
    return opt, NLopt.optimize(opt, x_in)
end
=#


#=

function minimize_pl_asym(alg::PlmAlg, var::PlmVar)

    LL = (var.N - 1) * var.q2 + var.q
    x0 = zeros(Float64, LL)
    g = zeros(Float64, LL)
    vecps = Vector{Float64}(undef, var.N)
    Jmat = zeros(LL, var.N)
    Threads.@threads for site = 1:var.N
        opt = Opt(alg.method, length(x0))
        ftol_abs!(opt, alg.epsconv)
        xtol_rel!(opt, alg.epsconv)
        xtol_abs!(opt, alg.epsconv)
        ftol_rel!(opt, alg.epsconv)
        maxeval!(opt, alg.maxit)
        min_objective!(opt, (x, g) -> optimfunwrapper(x, g, site, var))
        elapstime = @elapsed (minf, minx, ret) = optimize(opt, x0)
        alg.verbose && @printf("site = %d\t pl = %.4f\t time = %.4f\t", site, minf, elapstime)
        alg.verbose && println("exit status = $ret")
        vecps[site] = minf
        Jmat[:, site] .= minx
    end
    return Jmat, vecps
end


function pl_site_grad!(x::Vector{Float64}, grad::Vector{Float64}, site::Int, plmvar::PlmVar)
    LL = length(x)
    q2 = plmvar.q2
    q = plmvar.q
    N = plmvar.N
    M = plmvar.M
    Z = plmvar.Z
    W = plmvar.W
    IdxZ = plmvar.IdxZ
    for i = 1:LL-q
        grad[i] = 2.0 * plmvar.lambdaJ * x[i]
    end
    for i = (LL-q+1):LL
        grad[i] = 4.0 * plmvar.lambdaH * x[i]
    end
    pseudolike = 0.0
    vecene = zeros(Float64, q)
    lnorm = 0.0
    expvecenesumnorm = zeros(Float64, q)
    @inbounds for m in 1:M
        izm = view(IdxZ, :, m)
        zsm = Z[site, m]
        fillvecene!(vecene, x, site, izm, q, N)
        lnorm = logsumexp(vecene)
        expvecenesumnorm .= @. exp(vecene - lnorm)
        pseudolike -= W[m] * (vecene[zsm] - lnorm)
        @turbo for i in 1:site-1
            for s = 1:q
                grad[izm[i]+s] += W[m] * expvecenesumnorm[s]
            end
            grad[izm[i]+zsm] -= W[m]
        end
        @turbo for i = site+1:N
            for s = 1:q
                grad[izm[i]-q2+s] += W[m] * expvecenesumnorm[s]
            end
            grad[izm[i]-q2+zsm] -= W[m]
        end
        @turbo for s = 1:q
            grad[(N-1)*q2+s] += W[m] * expvecenesumnorm[s]
        end
        grad[(N-1)*q2+zsm] -= W[m]
    end
    pseudolike += l2norm_asym(x, plmvar)
end

=#