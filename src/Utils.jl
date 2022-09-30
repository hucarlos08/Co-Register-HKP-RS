module Utils

using LinearAlgebra
using StaticArrays
using Random
using LoopVectorization
using Images
using ImageDraw

export image_spatially_varying, upgrade_intensity, create_mosaic_view, draw_corners, draw_corners!, same_size_gray


function image_spatially_varying(image::Matrix{T}, K::Int64, scale::T, σₖ::T) where{T<:AbstractFloat}

    out::Matrix{T} = zeros(size(image))

    # Get image size
    w::Int64, h::Int64 = size(image)

    # Create random centers

    scale_w::Int64 = div(w,K)
    scale_h::Int64 = div(h,K)

    center_x::Vector{T} = [rand(((i*scale_w)+1):((i+1)*scale_w)) for i ∈ 0:K-1]
    center_y::Vector{T} = [rand(((i*scale_h)+1):((i+1)*scale_h)) for i ∈ 0:K-1]

    centers::SMatrix{2,K} = SMatrix{2,K}([center_x center_y])

    return image_spatially_varying(image, K, scale, σₖ, centers )
end

function image_spatially_varying(image::Matrix{T}, K::Int64, scale::T, σₖ::T, centers::AbstractArray{T}) where{T<:AbstractFloat}

    out::Matrix{T} = zeros(size(image))

    static_centers::SMatrix{2,K} =  SMatrix{2,K}(centers)

    # Create covariance matrix
    Σ¹::SMatrix{2,2} = SMatrix{2,2}(Diagonal([1.0/σₖ^2, 1.0/σₖ^2]))

    @simd for I ∈ CartesianIndices(image)

        x, y    = I.I
        diff    = static_centers .- SVector{2}([x, y]) 
        out[I]  = image[I] * (scale + (1.0/K) * sum(Diagonal(exp.(-0.5 * diff' * Σ¹ * diff))))

    end

    return out
end


function upgrade_intensity(pixel::T, scale::T=1.35) where {T<:AbstractFloat}
    return (1.0-pixel)^(scale)
end


function create_mosaic_view(image1::Matrix{T}, image2::Matrix{T}, wsize::Tuple{Int,Int}) where{T<:AbstractFloat}
    
    @assert size(image1) == size(image2)

    output::Matrix{T} = zeros(T, size(image1))

    rows, cols = size(image1)

    rowStep, r = divrem(rows, wsize[1])
    colStep, r = divrem(cols, wsize[2])

    iter::Int64 = 0
    for j ∈ 1:wsize[2]
        for i ∈ 1:wsize[1]

            rowInit  = (i-1)*rowStep + 1 
            rowFinal = min(i*rowStep, rows)

            
            colInit  = (j-1)*colStep + 1
            colFinal = min(j*colStep, cols) 

            if i%2 * j%2 == 0
                output[rowInit:rowFinal, colInit:colFinal] = image1[rowInit:rowFinal, colInit:colFinal]
            else
                output[rowInit:rowFinal, colInit:colFinal] = image2[rowInit:rowFinal, colInit:colFinal]
            end

            iter+=1

        end
    end

    return output

end


function create_mosaic_view(image1::Array{RGB{Normed{UInt8, 8}}, 2}, image2::Array{RGB{Normed{UInt8, 8}}, 2}, wsize::Tuple{Int,Int})
    
    @assert size(image1) == size(image2)

    output::Array{RGB{Normed{UInt8, 8}}, 2} = zeros(RGB{Normed{UInt8, 8}}, size(image1))

    rows, cols = size(image1)

    rowStep, r = divrem(rows, wsize[1])
    colStep, r = divrem(cols, wsize[2])

    for j ∈ 1:wsize[2]
        for i ∈ 1:wsize[1]

            rowInit  = (i-1)*rowStep + 1 
            rowFinal = min(i*rowStep, rows)

            
            colInit  = (j-1)*colStep + 1
            colFinal = min(j*colStep, cols) 

            if i%2 * j%2 == 0
                output[rowInit:rowFinal, colInit:colFinal] = image1[rowInit:rowFinal, colInit:colFinal]
            else
                output[rowInit:rowFinal, colInit:colFinal] = image2[rowInit:rowFinal, colInit:colFinal]
            end

        end
    end

    return output

end


function draw_corners(input::AbstractMatrix, corners::AbstractMatrix, c::Color, s::Int64=3, shape::String="o")

    @assert size(corners, 2) == 2

    output = copy(input)

    for i in 1:size(corners, 1)
        x, y = floor.(Int64, corners[i,:])
        if shape == "o"
            draw!(output, Ellipse(CirclePointRadius(x, abs(y), s)), c)
        else
            draw!(output, Cross(Point(x, abs(y)), s), c)
            
        end
    end
    
    return output
end

function draw_corners!(input::AbstractMatrix, corners::AbstractMatrix, c::Color, s::Int64=3, shape::String="o")

    @assert size(corners, 2) == 2

    for i in 1:size(corners, 1)
        x, y = floor.(Int64, corners[i,:])
        if shape == "o"
            draw!(input, Ellipse(CirclePointRadius(x, abs(y), s)), c)
        else
            draw!(input, Cross(Point(x, abs(y)), s), c)
            
        end
    end
    
    return input
end


function same_size_gray(image1::AbstractArray, image2::AbstractArray)
    
    if size(image1) == size(image2)
        return Gray.(image1), Gray.(image2)
    else
        rows = min(size(image1)[1], size(image2)[1])
        cols = min(size(image1)[2], size(image2)[2])

        return Gray.(image1[1:rows, 1:cols]), Gray.(image2[1:rows, 1:cols])
    end

end


end # end module
