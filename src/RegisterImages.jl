module RegisterImages

using Interpolations
using StaticArrays
using CoordinateTransformations
using ImageTransformations


import Base: size, minmax, vec, length
import Images: warp, clamp01nan

using Auxiliary

export Image,
    InterpolatedImage,
    size,
    minmax,
    warp,
    warp!,
    length,
    create_perspective_from_params,
    warpPerspective!,
    get_affin_from_params,
    get_inverse_affin

export AbstractTransformation,
    Affine,
    Perspective,
    vector_to_perspective_form,
    convert_affine_to_homogeneous,
    projec_points_from_affin,
    warp_multiple_channels


# "No threading nor vectorization."
abstract type AbstractTransformation end
struct Affine <: AbstractTransformation end
struct Perspective <: AbstractTransformation end


struct Image{T<:AbstractFloat}

    # Image
    pixels::Matrix{T}

    # Dynamic range
    min::T
    max::T

    function Image(image::Matrix{T}) where {T<:AbstractFloat}

        pixels = image
        min, max = findMinMax(vec(image))
        new{T}(pixels, min, max)

    end

end

@inline function size(img::Image{T}) where {T<:AbstractFloat}
    return Base.size(img.pixels)
end

@inline function minmax(img::Image{T}) where {T<:AbstractFloat}
    return (img.min, img.max)
end

@inline function vec(img::Image{T}) where {T<:AbstractFloat}
    return vec(img.pixels)
end

@inline function length(img::Image{T}) where {T<:AbstractFloat}
    return Base.length(img.pixels)
end

# ========================================= INTERPLATED IMAGE =============================================

struct InterpolatedImage{T<:AbstractFloat}

    # Basic Image data
    image::Image{T}

    # Interpoltion object
    itp::AbstractInterpolation

    # Memory for results
    pixels_interpolated::Matrix{T}

    function InterpolatedImage(img::Matrix{T}) where {T<:AbstractFloat}

        # Create basic info
        image = Image(img)

        # Create interpolation objetct

        itp = interpolate(img, BSpline(Cubic(Periodic(OnGrid()))))
        #itp = interpolate(img, BSpline(Linear()), OnGrid())

        # Create memory results
        pixels_interpolated = zeros(T, Base.size(img))

        new{T}(image, itp, pixels_interpolated)
    end

end

@inline function size(img::InterpolatedImage{T}) where {T<:AbstractFloat}
    return size(img.image)
end

@inline function minmax(img::InterpolatedImage{T}) where {T<:AbstractFloat}
    return minmax(img.image)
end

@inline function vec(img::InterpolatedImage{T}) where {T<:AbstractFloat}
    return vec(img.pixels_interpolated)
end

@inline function length(img::InterpolatedImage{T}) where {T<:AbstractFloat}
    return length(img.image)
end

# ========================================= AFFINE TRANSFORMATIONS =============================================


function get_affin_from_params(params::Vector{T}, offset::Vector{T}) where {T<:AbstractFloat}

    θ::T = params[1]
    sx::T = params[2]
    sy::T = params[3]
    tx::T = params[4]
    ty::T = params[5]

    off::SVector{2,T} = @SVector[offset[1], offset[2]]
    tras::SVector{2,T} = @SVector[tx, ty]

    rot::SMatrix{2,2,T} = SMatrix{2,2}([
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ])

    scale::SMatrix{2,2,T} = SMatrix{2,2}([sx, zero(T), zero(T), sy])

    M::SMatrix{2,2,T} = scale * rot
    t::SVector{2,T} = off - M * off + tras

    return M, t

end

@inline function get_inverse_affin(M::SMatrix{2,2,T}, t::SVector{2,T}) where {T<:AbstractFloat}

    # Cosntruct the inverse transform
    Mi::SMatrix{2,2,T} = inv(M)
    ti::SVector{2,T} = -Mi * t

    return Mi, ti

end

@inline function get_inverse_affin(image::Image{T}, params::Vector{T}) where {T<:AbstractFloat}

    off_x::T, off_y::T = ImageTransformations.center(image.pixels)

    M::SMatrix{2,2,T}, t::SVector{2,T} = get_affin_from_params(params, [off_x, off_y])

    Mi::SMatrix{2,2,T}, ti::SVector{2,T} = get_inverse_affin(M, t)

    return Mi, ti
end

@inline function get_inverse_affin(img::InterpolatedImage{T}, params::Vector{T}) where {T<:AbstractFloat}
    return get_inverse_affin(img.image, params)
end

function convert_affine_to_homogeneous(M::AbstractMatrix{T}, t::AbstractVector{T}) where {T<:AbstractFloat}

    return SMatrix{2,3,T}([
        M[1, 1] M[1, 2] t[1]
        M[2, 1] M[2, 2] t[2]
    ])

end


function warp!(img::InterpolatedImage{T}, params::Vector{T}, defaultValue::T, ::SIMD) where {T<:AbstractFloat}

    rows::Int64, cols::Int64 = size(img)

    Mi::SMatrix{2,2,T}, ti::SVector{2,T} = get_inverse_affin(img, params)

    # Create pointers to properties
    out::Matrix{T} = img.pixels_interpolated
    itp = img.itp

    @simd for I in CartesianIndices(axes(out))
        x, y = trunc.(Int64, Mi * SVector(I.I) + ti)
        if 1 <= x <= rows && 1 <= y <= cols
            @inbounds out[I] = itp(x, y)
        else
            @inbounds out[I] = defaultValue
        end
    end

    return img
end

function warp!(img::InterpolatedImage{T}, params::Vector{T}, defaultValue::T, ::PrivateThreads) where {T<:AbstractFloat}

    rows::Int64, cols::Int64 = size(img)

    Mi::SMatrix{2,2,T}, ti::SVector{2,T} = get_inverse_affin(img, params)

    # Create pointers to properties
    out::Matrix{T} = img.pixels_interpolated
    itp = img.itp

    Threads.@threads for I in CartesianIndices(axes(out))
        x, y = trunc.(Int64, Mi * SVector(I.I) + ti)
        if 1 <= x <= rows && 1 <= y <= cols
            @inbounds out[I] = itp(x, y)
        else
            @inbounds out[I] = defaultValue
        end
    end

    return img
end


function warp(image::Matrix{T}, params::Vector{T}, defaultValue::T, mode::Mode) where {T<:AbstractFloat}

    # Create interpolated memory
    itp_image::InterpolatedImage{T} = InterpolatedImage(image)
    warp!(itp_image, params, defaultValue, mode)

    return itp_image.pixels_interpolated
end

function warp_multiple_channels(image::AbstractArray, params::Vector{T}, defaultValue::T) where {T<:AbstractFloat}

    rows::Int64, cols::Int64 = size(image)

    # Get affin transformation
    off_x, off_y = center(image)

    # Get the esimated affin transformation
    M, t    = get_affin_from_params(params, [off_x, off_y]);
    Mi, ti  = get_inverse_affin(M, t);

    itp = interpolate(image, BSpline(Linear()), OnGrid())

    imw = similar(image, eltype(itp))


    Threads.@threads for I in CartesianIndices(axes(image))
        x, y = trunc.(Int64, Mi * SVector(I.I) + ti)
        if 1 <= x <= rows && 1 <= y <= cols
            @inbounds imw[I] = itp(x, y)
        else
            @inbounds imw[I] = clamp01nan(defaultValue)
        end
    end

    return imw

end

# ====================================== INVERSE MAPPING ===============================================

function inverseMapping!(
    out::Matrix{T},
    Hi::Matrix{T},
    itp::AbstractInterpolation,
    defaultValue::T,
    ::NoParallelization,
) where {T<:AbstractFloat}

    rows::Int64, cols::Int64 = size(out)

    for I in CartesianIndices(axes(out))
        x::Int64, y::Int64 = I.I
        x_new::Int64, y_new::Int64 = trunc.(Int64, Hi * SVector{3,T}(x, y, 1))

        if 1 <= x <= rows && 1 <= y <= cols
            @inbounds out[I] = itp(x_new, y_new)
        else
            @inbounds out[I] = defaultValue
        end
    end

end

function inverseMapping!(
    out::Matrix{T},
    Hi::Matrix{T},
    itp::AbstractInterpolation,
    defaultValue::T,
    ::SIMD,
) where {T<:AbstractFloat}

    rows::Int64, cols::Int64 = size(out)

    @simd for I in CartesianIndices(axes(out))
        x::Int64, y::Int64 = I.I
        x_new::Int64, y_new::Int64 = trunc.(Int64, Hi * SVector{3,T}(x, y, 1))

        if 1 <= x <= rows && 1 <= y <= cols
            @inbounds out[I] = itp(x_new, y_new)
        else
            @inbounds out[I] = defaultValue
        end
    end

end

function inverseMapping!(
    out::Matrix{T},
    Hi::Matrix{T},
    itp::AbstractInterpolation,
    defaultValue::T,
    ::PrivateThreads,
) where {T<:AbstractFloat}

    rows::Int64, cols::Int64 = size(out)

    Threads.@threads for I in CartesianIndices(axes(out))
        x::Int64, y::Int64 = I.I
        x_new::Int64, y_new::Int64 = trunc.(Int64, Hi * SVector{3,T}(x, y, 1))

        if 1 <= x <= rows && 1 <= y <= cols
            @inbounds out[I] = itp(x_new, y_new)
        else
            @inbounds out[I] = defaultValue
        end
    end

end

# ====================================== PERSPECTIVE PROJECTION ===============================================

function create_perspective_from_params(params::Vector{T}, offset::AbstractVector{T}) where {T<:AbstractFloat}

    θ::T, λₓ::T, λy::T, dx::T, dy::T, sx::T, sy::T = params

    Hr::SMatrix{3,3,T} = SMatrix{3,3,T}([
        cos(θ) -sin(θ) 0
        sin(θ) cos(θ) 0
        0.0 0.0 1
    ])

    Hs::SMatrix{3,3,T} = SMatrix{3,3,T}([
        λₓ 0 0
        0 λy 0
        0 0 1
    ])

    Hshe::SMatrix{3,3,T} = SMatrix{3,3,T}([1 sx 0; 0 1 0; 0 0 1]) * SMatrix{3,3,T}([1 0 0; sy 1 0; 0 0 1])

    M::SMatrix{3,3,T} = Hs * Hshe * Hr

    offset_::SVector{3,T} = SVector{3,T}((offset[1], offset[2], 1))

    t::SVector{3,T} = offset_ - Hr * offset_ + SVector{3,T}((dx, dy, 1))

    H::SMatrix{3,3,T} = SMatrix{3,3,T}([
        M[1, 1] M[1, 2] t[1]
        M[2, 1] M[2, 2] t[2]
        0.0 0.0 1.0
    ])

    return H

end


@inline function vector_to_perspective_form(x::Vector{T}) where {T<:AbstractFloat}

    return SMatrix{3,3,Float64}([
        x[1] x[2] x[3]
        x[4] x[5] x[6]
        0.0 0.0 1.0
    ])

end


function warpPerspective!(
    out::Matrix{T},
    H::SMatrix{3,3,T},
    itp::AbstractInterpolation,
    defaultValue::T,
    ::NoParallelization,
) where {T<:AbstractFloat}

    @assert (size(out) == size(itp))

    rows::Int64, cols::Int64 = size(out)

    # Get the inverse transforation
    Hi::SMatrix{3,3,T} = inv(H)

    for I in CartesianIndices(axes(out))

        x::Int64, y::Int64 = I.I

        x_new::Int64, y_new::Int64 = trunc.(Int64, Hi * SVector{3,T}(x, y, 1))

        if 1 <= x_new <= rows && 1 <= y_new <= cols
            @inbounds out[I] = itp(x_new, y_new)
        else
            @inbounds out[I] = defaultValue
        end
    end

    return out

end


function warpPerspective!(
    out::Matrix{T},
    H::SMatrix{3,3,T},
    itp::AbstractInterpolation,
    defaultValue::T,
    ::SIMD,
) where {T<:AbstractFloat}

    @assert (size(out) == size(itp))

    rows::Int64, cols::Int64 = size(out)

    # Get the inverse transforation
    Hi::SMatrix{3,3,T} = inv(H)

    @simd for I in CartesianIndices(axes(out))

        x::Int64, y::Int64 = I.I

        x_new::Int64, y_new::Int64 = trunc.(Int64, Hi * SVector{3,T}(x, y, 1))

        if 1 <= x_new <= rows && 1 <= y_new <= cols
            @inbounds out[I] = itp(x_new, y_new)
        else
            @inbounds out[I] = defaultValue
        end
    end

    return out

end


function warpPerspective!(
    out::Matrix{T},
    H::SMatrix{3,3,T},
    itp::AbstractInterpolation,
    defaultValue::T,
    ::PrivateThreads,
) where {T<:AbstractFloat}

    @assert (size(out) == size(itp))

    rows::Int64, cols::Int64 = size(out)

    # Get the inverse transforation
    Hi::SMatrix{3,3,T} = inv(H)

    Threads.@threads for I in CartesianIndices(axes(out))

        # Get coordinates
        x::Int64, y::Int64 = I.I
        # Estimate new coordinates 
        x_new::Int64, y_new::Int64 = trunc.(Int64, Hi * SVector{3,T}(x, y, 1))

        if 1 <= x_new <= rows && 1 <= y_new <= cols
            @inbounds out[I] = itp(x_new, y_new)
        else
            @inbounds out[I] = defaultValue
        end
    end

    return out

end

function warpPerspective!(
    img::InterpolatedImage{T},
    input::Vector{T},
    defaultValue::T,
    mode::Mode,
) where {T<:AbstractFloat}

    # Create pointers to properties
    out::Matrix{T} = img.pixels_interpolated
    itp = img.itp

    H::SMatrix{3,3,T} = vector_to_perspective_form(input)
    #H::SMatrix{3,3,T} = create_perspective_from_params(input, center(out))

    warpPerspective!(out, H, itp, defaultValue, mode)

    return img

end


function projec_points_from_affin(H::AbstractMatrix{T}, points::AbstractMatrix{T}) where {T<:AbstractFloat}

    count_points::Int64 = size(points, 1)
    projected_points::Matrix{T} = zeros(T, count_points, 2)

    @inbounds for i ∈ 1:count_points

        # True coordinates
        x, y = points[i, :]

        # Estimated coordinates
        x_est, y_est = H * SVector((x, y, 1))

        projected_points[i, 1] = x_est
        projected_points[i, 2] = y_est

    end

    return projected_points

end


end # end module
