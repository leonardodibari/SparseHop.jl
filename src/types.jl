struct Chain
    seq::Array{<:Integer,1}
    L::Int
    generator
end

function Chain(seq::Array{<:Integer,1}, generator)
    L = size(seq,1)
    Chain(seq, L, generator)
end

