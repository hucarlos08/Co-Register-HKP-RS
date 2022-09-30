module RegisterHistogram

using LoopVectorization
using Auxiliary

import Base: fill!

export  Histogram,
        fill_Histogram!,
        unpack_parameters

struct Histogram{T<:AbstractFloat}

    weights1::Vector{T}
    weights2::Vector{T}

    joint::Matrix{T}

    bins::Int64

    edges1::Tuple{T, T}
    edges2::Tuple{T, T}

    # Auxiliary memory
    weights1_subweights::Matrix{T}
    weights2_subweights::Matrix{T}
    joint_subweights::Array{T,3}

    function Histogram(nbins::Int64, edges1::Tuple{T,T}, edges2::Tuple{T,T}) where {T<:AbstractFloat}

        weights1    = zeros(T, nbins+1)
        weights2    = zeros(T, nbins+1)

        joint       = zeros(T, nbins+1, nbins+1)

        bins        = nbins

        edges1      = edges1
        edges2      = edges2

        weights1_subweights = zeros(T, nbins+1, Threads.nthreads());
        weights2_subweights = zeros(T, nbins+1, Threads.nthreads());
        joint_subweights    = zeros(T, nbins+1, nbins+1, Threads.nthreads());

        new{T}(weights1, weights2, joint, bins, edges1, edges2, weights1_subweights, weights2_subweights, joint_subweights)

    end

end

function clean_weights!(histogram::Histogram{T}) where {T<:AbstractFloat}
    
    x::T = zero(T)

    # Clean weights
    @turbo for i in eachindex(histogram.weights1)
        @inbounds histogram.weights1[i]    = x
        @inbounds histogram.weights2[i]    = x
    end

    @turbo for i in eachindex(histogram.joint)
        histogram.joint[i] = x
    end

    return histogram
end


function clean_subweights!(histogram::Histogram{T}) where {T<:AbstractFloat}
    
    x::T = zero(T)

    # Clean sub matrixs
    @turbo for i in eachindex(histogram.weights1_subweights)
        @inbounds histogram.weights1_subweights[i]    = x
        @inbounds histogram.weights2_subweights[i]    = x
    end

    # Clean joint subweights
    @turbo for i in eachindex(histogram.joint_subweights)
        @inbounds histogram.joint_subweights[i] = x
    end

    return histogram
end


function unpack_parameters(histogram::Histogram{T}) where{T<:AbstractFloat}
    
    xmin::T, xmax::T = histogram.edges1
    ymin::T, ymax::T = histogram.edges2

    bins::Int64 = histogram.bins

    norm_x::T = bins/(xmax - xmin)
    norm_y::T = bins/(ymax - ymin)

    return bins, (xmin, norm_x), (ymin, norm_y)
end

function sum_subweights!(histogram::Histogram{T}) where{T<:AbstractFloat}
    
    joint::Matrix{T}    = histogram.joint
    weights1::Vector{T} = histogram.weights1
    weights2::Vector{T} = histogram.weights2

    weights1_subweights::Matrix{T}  = histogram.weights1_subweights
    weights2_subweights::Matrix{T}  = histogram.weights2_subweights
    joint_subweights::Array{T,3}    = histogram.joint_subweights

    # Sum individual histograms
    @turbo for j in axes(weights1_subweights,2)
        for i in axes(weights1_subweights,1)

            weights1[i] += weights1_subweights[i,j] 
            weights2[i] += weights2_subweights[i,j]
        end
    end

   @turbo sum!(joint, joint_subweights)

end

function fill_Histogram!(histogram::Histogram{T}, x::Vector{T}, y::Vector{T}, ::NoParallelization) where{T<:AbstractFloat}

    # Chech size
    @assert(size(x)==size(y))

    # Clean previus values
    clean_weights!(histogram)

    bins::Int64, (xmin::T, norm_x::T), (ymin::T, norm_y::T) = unpack_parameters(histogram)

    weights1::Vector{T} = histogram.weights1
    weights2::Vector{T} = histogram.weights2
    joint::Matrix{T}    = histogram.joint

    # Fill weights
   for i in eachindex(x)
        index_x = getbin(x[i], xmin, norm_x, bins)
        index_y = getbin(y[i], ymin, norm_y, bins)

        weights1[index_x] += 1
        weights2[index_y] += 1

        joint[index_x, index_y] += 1
    end

    return histogram
end


function fill_Histogram!(histogram::Histogram{T}, x::Vector{T}, y::Vector{T}, ::SIMD) where{T<:AbstractFloat}

    # Chech size
    @assert(size(x)==size(y))

    # Clean previus values
    clean_weights!(histogram)

    bins::Int64, (xmin::T, norm_x::T), (ymin::T, norm_y::T) = unpack_parameters(histogram)

    weights1::Vector{T} = histogram.weights1
    weights2::Vector{T} = histogram.weights2
    joint::Matrix{T}    = histogram.joint

    # Fill weights
    @simd for i in eachindex(x)
        @inbounds index_x = getbin(x[i], xmin, norm_x, bins)
        @inbounds index_y = getbin(y[i], ymin, norm_y, bins)

        @inbounds weights1[index_x] += 1
        @inbounds weights2[index_y] += 1

        @inbounds joint[index_x, index_y] += 1
    end

    return histogram
end

function fill_Histogram!(histogram::Histogram{T}, x::Vector{T}, y::Vector{T}, ::PrivateThreads) where{T<:AbstractFloat}
    
    # Chech size
    @assert(size(x)==size(y))

    # Clean previus values
    clean_weights!(histogram)
    clean_subweights!(histogram)

    bins::Int64, (xmin::T, norm_x::T), (ymin::T, norm_y::T) = unpack_parameters(histogram)

    nthreads::Int64    = Threads.nthreads()
    k_inputs::Int64,   = size(x)
    Threads.@threads for thread_idx = 1:nthreads

        weights1_idx    = @view histogram.weights1_subweights[:, thread_idx]
        weights2_idx    = @view histogram.weights2_subweights[:, thread_idx]
        joint_idx       = @view histogram.joint_subweights[:, :, thread_idx]

        # Fill weights
        @simd for i in get_range(k_inputs)

            @inbounds index_x = getbin(x[i], xmin, norm_x, bins)
            @inbounds index_y = getbin(y[i], ymin, norm_y, bins)

            @inbounds weights1_idx[index_x]   += 1
            @inbounds weights2_idx[index_y]   += 1

            @inbounds joint_idx[index_x, index_y] += 1
        end
    end

    # add all
    sum_subweights!(histogram)

    return histogram

end
    
end #end module