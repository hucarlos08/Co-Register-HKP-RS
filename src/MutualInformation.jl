module MutualInformation

using LoopVectorization

using RegisterHistogram
using RegisterImages
using RegisterMemory

using Auxiliary


export  estimate_similarity_MI!



function estimate_similarity_MI!(memory::Memory{T}, params::Vector{T}, simType::SIMType, mode::Mode) where{T<:AbstractFloat}
    
    # Apply affine transformation
    warp!(memory, params, mode)
    
    fill_Histogram!(memory, mode)

    return estimate_similarity_MI!(memory, simType)
end

@inline function estimate_similarity_MI!(memory::Memory{T}, simType::SIMType) where{T<:AbstractFloat}
    return estimate_similarity_MI(memory.histogram.joint, simType)
end

function estimate_similarity_MI(joint::Matrix{T}, simType::SIMType) where {T<:AbstractFloat}

    a::Matrix{T} = @view joint[1:end-1,1:end-1]

    b::Matrix{T}        = a./sum(a)         # normalized joint histogram
    y_marg::Matrix{T}   = sum(b, dims=2)    # sum of the rows of normalized joint histogram
    x_marg::Matrix{T}   = sum(b, dims=1)    # sum of columns of normalized joint histogran

    Hy::T = 0.0
    @simd for i in eachindex(y_marg)    #  columns
        if(!iszero(y_marg[i]))
            Hy += -y_marg[i]*log2(y_marg[i]) # marginal entropy for image 1
        end
    end

    Hx::T = 0.0
    @simd for i in eachindex(x_marg)   #rows
        if(!iszero(x_marg[i]))
            Hx += -x_marg[i]*log2(x_marg[i]) # marginal entropy for image 2
        end
    end

    H_xy::T = 0.0
    @simd for i in eachindex(b)
        if(!iszero(b[i]))
            H_xy += -b[i]*log2(b[i])
        end
    end

    return similarityMI(Hx, Hy, H_xy, simType)
end

@inline function similarityMI(Hx::T, Hy::T, H_xy::T, ::MI) where{T<:AbstractFloat}
    maximum::T = 2.0
    if !iszero(H_xy)
        return -((Hx + Hy)/H_xy) # Normalized Mutual information
    else
        return maximum
    end
end


end #end module
