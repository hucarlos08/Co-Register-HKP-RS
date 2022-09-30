module HistogramKP

using LoopVectorization

using RegisterHistogram
using RegisterImages
using RegisterMemory

using Auxiliary

export  estimate_similarity_HKP!,
        HKPType,
        HKPAdd,
        HKP



function estimate_similarity_HKP!(memory::Memory{T}, params::Vector{T}, simType::SIMType, mode::Mode) where{T<:AbstractFloat}
    
    # Apply affine transformation
    warp!(memory, params, mode)

    # Histogram estimation
    fill_Histogram!(memory, mode)

    return estimate_similarity_HKP!(memory, simType)

end

@inline function estimate_similarity_HKP!(memory::Memory{T}, simType::SIMType) where{T<:AbstractFloat}
        return estimate_similarity_HKP(memory.histogram, length(memory), simType)
end


@inline function estimate_similarity_HKP(histogram::Histogram{T}, N, simType::SIMType) where{T<:AbstractFloat}
    
    # Reference to memory
    weights1::Vector{T} = histogram.weights1
    weights2::Vector{T} = histogram.weights2
    joint::Matrix{T}    = histogram.joint

    return similarityHKP(weights1, weights2, joint, histogram.bins, N, simType)
end


function similarityHKP(
    weights1::Vector{T},
    weights2::Vector{T},
    joint::Matrix{T},
    bins::Int64,
    N::Int64,
    simType::SIMType) where {T<:AbstractFloat}

    result_reference::T    = 0.0
    result_target::T       = 0.0
    result_joint::T        = 0.0

    @turbo for j = 1:bins

        @inbounds result_target       += (weights1[j] * weights1[j]) - weights1[j]
        @inbounds result_reference    += (weights2[j] * weights2[j]) - weights2[j]

        for i = 1:bins
            @inbounds result_joint += (joint[i, j] * joint[i, j]) - joint[i, j]
        end
    end

    hkp_reference::T   = result_reference / N
    hkp_target::T      = result_target / N
    hkp_joint::T       = result_joint / N

    return SimilarityHKP(hkp_target, hkp_reference, hkp_joint, simType)
end


@inline function SimilarityHKP(kpR::T, kpT::T, kpJ::T, ::HKPAdd) where{T<:AbstractFloat}
    return kpT + kpR - (kpJ + kpJ)
end

@inline function SimilarityHKP(kpR::T, kpT::T, kpJ::T, ::HKP) where{T<:AbstractFloat}
    maximum::T      = 0.5
    sum_marginal::T = kpT + kpR

    if !iszero(sum_marginal)
        return   - (kpJ / sum_marginal)#maximum - (kpJ / sum_marginal)
    else
        return maximum
    end
end


end # end module