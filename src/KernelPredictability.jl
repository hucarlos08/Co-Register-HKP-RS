
module KernelPredictability

using Base: eltypeof
using LinearAlgebra
using Interpolations
using CoordinateTransformations
using StaticArrays
using AffineTransformation
using Random
using LoopVectorization

export  gaussian_kernel,
        KP₁,
        KP₂,
        KP₃,
        idKernel,
        idSimilarity,
        similarityKP

function gaussian_kernel(xᵢ, xⱼ, γ)

    out = exp(-γ*dot(xᵢ-xⱼ,xᵢ-xⱼ))
    out = isnan(out) ? 0.0 : out

    return out

end

function KP₁(samplesT::AbstractArray, samplesR::AbstractArray, γ::Float64)

    kpT = 0.0
    kpR = 0.0
    kpJ = 0.0

    n, = size(samplesT)

    for i in 1:(n-1)
        @avx  for j in (i+1):n
            δ1 = samplesT[i]-samplesT[j]
            δ2 = samplesR[i]-samplesR[j]

            δ1 = isnan(δ1) ? Inf : δ1
            δ2 = isnan(δ1) ? Inf : δ2

            kpT += exp(-γ*δ1*δ1)
            kpR += exp(-γ*δ2*δ2)
            kpJ += exp(-γ*(δ1*δ1+δ2*δ2))

        end

    end

    return kpT, kpR, kpJ

end

function KP₂(samplesT::AbstractArray,samplesR::AbstractArray, γ::Float64)

    kpT = 0.0
    kpR = 0.0
    kpJ = 0.0

    n, = size(samplesT)

    half = trunc(Int64, n/2)

    @avx for i in eachindex(samplesT[1:half])
        for  j in (half+1):n
            δ1 = samplesT[i]-samplesT[j]
            δ2 = samplesR[i]-samplesR[j]

            δ1 = isnan(δ1) ? Inf : δ1
            δ2 = isnan(δ1) ? Inf : δ2

            kpT += exp(-γ*δ1*δ1)
            kpR += exp(-γ*δ2*δ2)
            kpJ += exp(-γ*(δ1*δ1+δ2*δ2))

        end

    end

    return kpT, kpR, kpJ


end


function KP₃(samplesT::AbstractArray, samplesR::AbstractArray, γ::Float64)

    kpT = 0.0
    kpR = 0.0
    kpJ = 0.0
    @avx for i in eachindex(samplesT)
        for j in eachindex(samplesT)

            δ1 = samplesT[i]-samplesT[j]
            δ2 = samplesR[i]-samplesR[j]

            δ1 = isnan(δ1) ? Inf : δ1
            δ2 = isnan(δ1) ? Inf : δ2

            kpT += exp(-γ*δ1*δ1)
            kpR += exp(-γ*δ2*δ2)
            kpJ += exp(-γ*(δ1*δ1+δ2*δ2))

        end

    end

    return kpT, kpR, kpJ
   
end

function idKernel(γ::Float64, idKP::Int64, samplesT::AbstractArray, samplesR::AbstractArray)

    kpT = 0.0
    kpR = 0.0
    kpJ = 0.0

    if idKP == 1

        kpT, kpR, kpJ = KP₁(samplesT, samplesR, γ)

    elseif idKP == 2

        kpT, kpR, kpJ = KP₂(samplesT, samplesR, γ)

    elseif idKP == 3

        kpT, kpR, kpJ = KP₃(samplesT, samplesR, γ)
    end

    return kpT, kpR, kpJ
end

function idSimilarity(idSim::Int64, kpT::Float64, kpR::Float64, kpJ::Float64)

    if idSim == 1
        return -kpJ/(kpT + kpR)

    elseif idSim == 2
        return kpT + kpR - (2.0*kpJ)

    end

end


function similarityKP(
    params::AbstractArray,
    reference::AbstractMatrix,
    target::AbstractMatrix,
    γ::Float64,
    nSamples::Int64,
    itp::AbstractInterpolation,
    idKP::Int64,
    idSim::Int64,
    rng::MersenneTwister = MersenneTwister(0)
)

    rows, cols = size(target)

    off_x = convert(eltype(params), rows / 2.0)
    off_y = convert(eltype(params), cols / 2.0)

    M, t = affineT(params, [off_x, off_y])

    # Cosntruct the inverse transform
    Mi = inv(M)
    ti = -Mi * t

    tform = AffineMap(Mi, ti)

    inf = convert(eltype(params), Inf)

    # Estimate samples over traslape
    count    = 0

    out_target      = fill(inf, nSamples)
    out_reference   = fill(inf, nSamples)

    # shuffle pixels
    pixels = shuffle(rng, collect(CartesianIndices(axes(reference))))

    count = 1

    for I ∈ pixels
        x, y    = trunc.(Int64, tform(SVector(I.I)))

        if (1<=x<=rows) && (1<=y<=cols)
            @inbounds out_target[count]       = target[I]
            @inbounds out_reference[count]    = itp(x,y)

            count += 1
        else
            @inbounds out_target[count]     = inf
            @inbounds out_reference[count]  = inf
        end

        if count > nSamples
            break
        end

    end

    kpT, kpR, kpJ = idKernel(γ, idKP, out_target, out_reference)

    return idSimilarity(idSim, kpT, kpR, kpJ)

end

end # end module
