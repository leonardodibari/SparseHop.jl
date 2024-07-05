function count_tuple(v, n)
    d = Dict{Int, Int}()
    for l in 1:n
        i = v[1][l]
        if haskey(d,i)
            d[i] += 1
        else
            d[i] = 1
        end
    end
    
    for j in 2:length(v)
        for l in 1:n
            i = v[j][l]
            if haskey(d,i)
                d[i] += 1
            else
                d[i] = 1
            end
        end
    end
    return d
end
    
        
        