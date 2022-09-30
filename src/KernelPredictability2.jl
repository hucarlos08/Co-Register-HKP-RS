module KernelPredictability2

using Random
using LinearAlgebra
using LoopVectorization

using RegisterHistogram
using RegisterImages
using RegisterMemory

using Auxiliary

export estimate_similarity_KP


@inline function gaussian_kernel(xᵢ::T, xⱼ::T, γ::T) where {T<:AbstractFloat}

    out::T = exp(-γ * dot(xᵢ - xⱼ, xᵢ - xⱼ))
    return isnan(out) ? 0.0 : out

end

function estimate_similarity_KP(memory::Memory{T}, params::Vector{T}, simType::SIMType, mode::Mode) where{T<:AbstractFloat}
    
    # Apply affine transformation
    warp!(memory.reference, params, memory.default_pixel, mode)

    # Get traslaped pixels
    nSamples::Int64 = 500
    γ::T = 10
    pixels_target, pixels_reference = get_traslape_pixels(memory, nSamples)

    return estimate_similarity_KP(pixels_target, pixels_reference, γ, simType)

end

function get_traslape_pixels(memory::Memory{T}, nSamples::Int64) where{T<:AbstractFloat}
    
    image_target::Matrix{T}    = memory.target.pixels
    image_reference::Matrix{T} = memory.reference.pixels_interpolated

    # shuffle pixels
    suffled_indexes = shuffle(collect(CartesianIndices(axes(image_target))))

    # Serch for pixels
    count::Int64                = 1
    pixels_target::Vector{T}    = Vector{T}(undef, nSamples)
    pixels_reference::Vector{T} = Vector{T}(undef, nSamples)
    inf::T                      = convert(T, Inf)

    for I ∈ suffled_indexes

        if !isnan(image_reference[I])
            @inbounds pixels_target[count]      = image_target[I]
            @inbounds pixels_reference[count]   = image_reference[I]
            count += 1
        else
            @inbounds pixels_target[count]      = inf
            @inbounds pixels_reference[count]   = inf
        end

        if count > nSamples
            break
        end
    end

    return pixels_target, pixels_reference
    
end

function estimate_similarity_KP(samplesT::Vector{T}, samplesR::Vector{T}, γ::T, simType::SIMType) where {T<:AbstractFloat}

    kpT::T = 0.0
    kpR::T = 0.0
    kpJ::T = 0.0
    @tturbo for i in eachindex(samplesT)
        for j in eachindex(samplesT)

            @inbounds δ1 = samplesT[i] - samplesT[j]
            @inbounds δ2 = samplesR[i] - samplesR[j]

            δ1 = isnan(δ1) ? Inf : δ1
            δ2 = isnan(δ1) ? Inf : δ2

            kpT += exp(-γ * δ1 * δ1)
            kpR += exp(-γ * δ2 * δ2)
            kpJ += exp(-γ * (δ1 * δ1 + δ2 * δ2))

        end

    end

    return SimilarityKP(kpT, kpR, kpJ, simType)

end



@inline function SimilarityKP(kpR::T, kpT::T, kpJ::T, ::KPAdd) where{T<:AbstractFloat}
    return kpT + kpR - (kpJ + kpJ)
end

@inline function SimilarityKP(kpR::T, kpT::T, kpJ::T, ::KPNormalized) where{T<:AbstractFloat}
    maximum::T      = 0.5
    sum_marginal::T = kpT + kpR

    if !iszero(sum_marginal)
        return  maximum - (kpJ / sum_marginal)
    else
        return maximum
    end
end


end # end module 