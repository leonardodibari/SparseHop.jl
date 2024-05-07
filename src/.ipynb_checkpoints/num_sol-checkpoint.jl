
function zero_eq!(x, idx, cost, p2, V, i, j, head, reg, q) #this withoun anything is 2 allocations, with everything 4 alloc, this function computes the y_k that should be 0 given a certain k and i,j,head
    x.vars .= 0  #this alone is 3 allocations
    for a in 1:q #this for alone is 3 allocations
        for b in 1:q
            x.vars[1] += exp(x.k[idx] * V[a,b,head])*V[a,b,head]*p2[a,i,b,j]
            x.vars[2] += exp(x.k[idx] * V[a,b,head])*p2[a,i,b,j]
        end
    end
    x.dest[idx] = cost - (x.vars[1]/x.vars[2]) - (2*reg*x.k[idx]) #4 allocations this line
end

function zero_eq2!(x, idx, cost, p2, V, i, j, head, reg, q)
    x.vars .= 0
end



function bisection!(k, y_k, dL, x, cost, f2, V, i, j, head, reg, q) ## computes k, y_k and dL from the self-consistent equation given a precise i,j,head
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
                x.vars[2] = 0
                for a in 1:q
                    for b in 1:q
                        x.vars[2] += exp(x.k[2] * V[a,b,head])*f2[a,i,b,j] 
                    end
                end
                dL[i,j,head] = cost - log(x.vars[2]) - (reg * x.k[2] *x.k[2])
                return
            else 
                k[i,j,head] = x.k[1]
                y_k[i,j,head] = x.dest[1]
                x.vars[2] = 0
                for a in 1:q
                    for b in 1:q
                        x.vars[2] += exp(x.k[1] * V[a,b,head])*f2[a,i,b,j] 
                    end
                end
                dL[i,j,head] = cost - log(x.vars[2]) - (reg * x.k[1] *x.k[1])
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
            x.vars[2] = 0
            for a in 1:q
                for b in 1:q
                    x.vars[2] += exp(x.k[3] * V[a,b,head])*f2[a,i,b,j] 
                end
            end
            dL[i,j,head] = cost - log(x.vars[2]) - (reg * x.k[3] *x.k[3])
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
    
    x.vars[2] = 0
    for a in 1:q
        for b in 1:q
            x.vars[2] += exp(x.k[3] * V[a,b,head])*f2[a,i,b,j] 
        end
    end
    
    dL[i,j,head] = cost - log(x.vars[2]) - (reg * x.k[3] *x.k[3])
end      




    
    


