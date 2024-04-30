function small_k(Df2, f2, V; reg = 0.1)
    @tullio Vm := V[a,b]*f2[a,b]
    @tullio num := V[a,b]*Df2[a,b]  
    @tullio den := V[a,b]*V[a,b]*f2[a,b]
    num = num - Vm
    den = den - (Vm)^2 + 2*reg
    return num/den,  medie(Df2, f2, num/den, V)
end

function dlog_small_k(Df2, f2, V; reg = 0.1)
    @tullio Vm := V[a,b]*f2[a,b]
    @tullio num := V[a,b]*Df2[a,b]  
    @tullio den := V[a,b]*V[a,b]*f2[a,b]
    num = (num - Vm)
    den = den - (Vm)^2 + 2*reg
    return num/den, (num^2)/den
end

function medie(Dp2, p2, k, V; reg = 0.1)
    @tullio num := exp(k * V[a,b])*V[a,b]*p2[a,b]
    @tullio den := exp(k * V[a,b])*p2[a,b] 
    @tullio cost := V[a,b]*Dp2[a,b] 
    return cost - (num/den) - (2*reg*k) 
end

function dlog(Dp2, p2, k, V; reg = 0.1)
    @tullio den := exp(k * V[a,b])*p2[a,b]
    @tullio cost := V[a,b]*Dp2[a,b]
    return k*cost - log(den) - (reg*k*k) 
end

function bisection(Df2, f2, V; tol = 1e-6, tol2 = 1e-10, reg = 0.1)
    Ka = -0.1
    Kb = 0.1
    
    while medie(Df2, f2, Ka, V, reg = reg) * medie(Df2, f2, Kb, V, reg = reg) > 0 
        Ka -= 5
        Kb += 5
        if Ka < -300 || Kb > 300
            if medie(Df2, f2, Ka, V, reg = reg) > medie(Df2, f2, Kb, V, reg = reg)
                return Ka, medie(Df2, f2, Kb, V)
            else 
                return Kb, medie(Df2, f2, Ka, V, reg = reg)
            end
        end
    end

    while (Kb - Ka) / 2 > tol
        
        Kc = (Ka + Kb) / 2
        if abs(medie(Df2, f2, Kc, V, reg = reg)) < tol2
            return Kc, medie(Df2, f2, Kc, V)
        elseif medie(Df2, f2, Ka, V, reg = reg) * medie(Df2, f2, Kc, V, reg = reg) < 0
            Kb = Kc
        else
            Ka = Kc
        end
    end
    
    return (Ka + Kb)/2 , medie(Df2, f2, (Ka+Kb)/2, V, reg = reg)

end      




    
    


