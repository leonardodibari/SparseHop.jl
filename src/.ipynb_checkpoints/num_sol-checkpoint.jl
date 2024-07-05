
function zero_eq!(x, idx::Int, cost::T, p2::Array{T,4}, V::Array{T,3}, i::Int, j::Int, head::Int, reg::T, q::Int) where {T} #this function computes the derivative of dLog given a certain k and i,j,head
    num = zero(T)
    den = zero(T)
    for a in 1:q 
        for b in 1:q
            num += exp(x.k[idx] * V[a,b,head])*V[a,b,head]*p2[a,i,b,j]
            den += exp(x.k[idx] * V[a,b,head])*p2[a,i,b,j]
        end
    end
    x.dest[idx] = cost - (num/den) - (2*reg*x.k[idx])
end

function zero_eq(k::T, cost::T, p2::Array{T,4}, V::Array{T,3}, i::Int, j::Int, head::Int, reg::T, q::Int) where {T} #this function computes the derivative of dLog given a certain k and i,j,head
    #x.vars .= T.(0.)
    num = zero(T)
    den = zero(T)
    for a in 1:q 
        for b in 1:q
            num += exp(k * V[a,b,head])*V[a,b,head]*p2[a,i,b,j]
            den += exp(k * V[a,b,head])*p2[a,i,b,j]
        end
    end
     cost - (num/den) - (2*reg*k)
end


function dlog(f2::Array{T,4}, Df2::Array{T,4}, k::T, V::Array{T,3}, i::Int, j::Int, head::Int, reg::T, q::Int) where {T}
    count = T(0.)
    count2 = T(0.)
    
    for a in 1:q
        for b in 1:q
            count2 += k * V[a,b,head]*Df2[a,i,b,j]
            count += exp(k * V[a,b,head])*f2[a,i,b,j] 
        end
    end
    
    return count2 - log(count) - (reg*k*k)
end    


function bisection!(k::Array{T,3}, y_k::Array{T,3}, x, cost::T, f2::Array{T,4}, V::Array{T,3}, i::Int, j::Int, head::Int, reg::T, q::Int) where {T} # finds the k which sets to 0 the derivative of dLog from the self-consistent equation given a precise i,j,head
    x.k[1] = -0.1
    x.k[2] = 0.1
    zero_eq!(x, 1, cost, f2, V, i, j, head, reg, q) 
    zero_eq!(x, 2, cost, f2, V, i, j, head, reg, q)
    
    while x.dest[1] * x.dest[2] > 0 #if i don't have sol in (a,b), enlarge the interval
        
        x.k[1] -= 5
        x.k[2] += 5
        
        zero_eq!(x, 1, cost, f2, V, i, j, head, reg, q) 
        zero_eq!(x, 2, cost, f2, V, i, j, head, reg, q)
        
        if x.k[1] < -100 || x.k[2] > 100   #stop when k is too big and give the extreme that has smaller y_k
            if x.dest[1] > x.dest[2]
                k[i,j,head] = x.k[2]
                y_k[i,j,head] = x.dest[2]
                return
            else 
                k[i,j,head] = x.k[1]
                y_k[i,j,head] = x.dest[1]
                return
            end
        end
    end

    while ((x.k[2] - x.k[1]) / 2) > 1e-6 #while (a,b) are far away continue to search for optimal solution
        x.k[3] = (x.k[1] + x.k[2]) / 2
        zero_eq!(x, 3, cost, f2, V, i, j, head, reg, q)  
        if abs(x.dest[3]) < 1e-10  #when y_k becomes very close to 0 stop
            k[i,j,head] = x.k[3]
            y_k[i,j,head] = x.dest[3]
            return
        elseif x.dest[1] * x.dest[3] < 0
            x.k[2] = x.k[3]
        else
            x.k[1] = x.k[3]
        end
        zero_eq!(x, 1, cost, f2, V, i, j, head, reg, q) 
    end
    
    #when (a,b) are close to each other then take the mean point as solution
    x.k[3] = (x.k[1]+x.k[2])/2
    zero_eq!(x, 3, cost, f2, V, i, j, head, reg, q) 
    
    k[i,j,head] = x.k[3]
    y_k[i,j,head] = x.dest[3]
end    


function get_dlog!(k::Array{T,3}, y_k::Array{T,3}, dL::Array{T,3}, V::Array{T,3}, D, M, str, reg::T, q::Int, L::Int, H::Int) where {T}
    for i in 1:L 
        for j in (i+1):L 
            @tasks for head in 1:H
                bisection!(k, y_k, str[head], D.mheads[i,j,head], M.f2rs, V, i, j, head, reg, q)  
            end 
        end
    end
                        
    for i in 1:L 
        for j in (i+1):L 
            @tasks for head in 1:H
                dL[i,j,head] = dlog(M.f2rs, D.f2rs, k[i,j,head], V, i, j, head, reg, q)
            end 
        end
    end
end
       
    
function activate_edges!(k::Array{T,3}, K::Array{T,3}, y_k::Array{T,3}, dL::Array{T,3}, graf, full_graf, verbose, avoid_upd, opt_k, n_edges::Int, history::Array{Int,3}, iter, order_list) where {T}
    
    L, L, H = size(K)
    all_edges = round(Int,L*(L-1)*H/2)
    
    copy_dL = deepcopy(dL)
    order = []
    for _ in 1:all_edges
        m, n, nu = Tuple(argmax(copy_dL))
        push!(order, Tuple(argmax(copy_dL)))
        copy_dL[m,n,nu] = 0
    end   
    push!(order_list, order)
    
    if avoid_upd == true
        dL[K .!= 0.] .= T(0.)
    end
            
    for _ in 1:n_edges
        m, n, nu = Tuple(argmax(dL))
        history[m,n,nu] = iter
        if verbose == true
            println(" Suggested K : $(k[m,n,nu]), grad_eq $(y_k[m,n,nu]) terms $(argmax(dL)), dL : $(maximum(dL))")           end
        if opt_k == true
            K[m,n,nu] = k[m,n,nu]
            K[n,m,nu] = K[m,n,nu]
        end
        add_edge!(graf[nu], m, n)
        add_edge!(full_graf, m, n)
        dL[m,n,nu] = T(0.)
    end
    dL .= 0
end
    