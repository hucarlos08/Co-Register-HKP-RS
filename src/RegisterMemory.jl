module RegisterMemory

using OffsetArrays
using Images

using RegisterHistogram
using RegisterImages
using RegisterFilters
using Auxiliary

import RegisterImages: warp!, length
import RegisterHistogram: fill_Histogram!

export AbstractMemory,
    Memory, MemoryGradient, length, warp!, fill_Histogram!, get_gradient!, sum_edges_angles, sum_exp_edges_angles

abstract type AbstractMemory{T<:AbstractFloat} end

struct Memory{T<:AbstractFloat}

    target::Image{T}
    reference::InterpolatedImage{T}

    histogram::Histogram{T}

    default_pixel::T

    function Memory(
        img_target::Matrix{T},
        img_reference::Matrix{T},
        bins::Int64,
        defaultpixel::T,
    ) where {T<:AbstractFloat}

        target = Image(img_target)
        reference = InterpolatedImage(img_reference)

        histogram = Histogram(bins, minmax(target), minmax(reference))

        default_pixel = defaultpixel

        new{T}(target, reference, histogram, default_pixel)
    end

end


@inline function length(memory::Memory{T}) where {T<:AbstractFloat}
    return RegisterImages.length(memory.target)
end

@inline function warp!(memory::Memory{T}, params::Vector{T}, mode::Mode) where {T<:AbstractFloat}
    RegisterImages.warp!(memory.reference, params, memory.default_pixel, mode)
end

@inline function fill_Histogram!(memory::Memory{T}, mode::Mode) where {T<:AbstractFloat}
    RegisterHistogram.fill_Histogram!(memory.histogram, vec(memory.target), vec(memory.reference), mode)
end

# ===================================================================================================
struct MemoryGradient{T<:AbstractFloat}

    memory::Memory{T}

    target_dx::OffsetArray{T}
    target_dy::OffsetArray{T}

    reference_dx::OffsetArray{T}
    reference_dy::OffsetArray{T}

    kernel_dx::AbstractMatrix{T}
    kernel_dy::AbstractMatrix{T}

    function MemoryGradient(
        img_target::Matrix{T},
        img_reference::Matrix{T},
        bins::Int64,
        defaultpixel::T,
        mode::Mode,
    ) where {T<:AbstractFloat}

        memory = Memory(img_target, img_reference, bins, defaultpixel)

        kernel_dy, kernel_dx = Images.Kernel.ando3()

        target_dx = OffsetArray(similar(img_target, size(img_target) .- size(kernel_dx) .+ 1), -1 .- kernel_dx.offsets)
        target_dy = OffsetArray(similar(img_target, size(img_target) .- size(kernel_dy) .+ 1), -1 .- kernel_dy.offsets)

        reference_dx =
            OffsetArray(similar(img_reference, size(img_reference) .- size(kernel_dx) .+ 1), -1 .- kernel_dx.offsets)
        reference_dy =
            OffsetArray(similar(img_reference, size(img_reference) .- size(kernel_dy) .+ 1), -1 .- kernel_dy.offsets)

        # Estimate the gradient for the target image

        filter2davx!(target_dx, img_target, kernel_dx, mode)
        filter2davx!(target_dy, img_target, kernel_dy, mode)

        new{T}(memory, target_dx, target_dy, reference_dx, reference_dy, kernel_dx, kernel_dy)
    end

end

@inline function length(memory_gradient::MemoryGradient{T}) where {T<:AbstractFloat}
    return length(memory_gradient.memory)
end

@inline function warp!(memory_gradient::MemoryGradient{T}, params::Vector{T}, mode::Mode) where {T<:AbstractFloat}
    warp!(memory_gradient.memory, params, mode)
end

@inline function fill_Histogram!(memory_gradient::MemoryGradient{T}, mode::Mode) where {T<:AbstractFloat}
    fill_Histogram!(memory_gradient.memory, mode)
end

@inline function get_gradient!(memory_gradient::MemoryGradient{T}, mode::Mode) where {T<:AbstractFloat}

    # Get the interpolated pixels
    image::Matrix{T} = memory_gradient.memory.reference.pixels_interpolated
    kernel_dx::AbstractMatrix{T} = memory_gradient.kernel_dx
    kernel_dy::AbstractMatrix{T} = memory_gradient.kernel_dy

    reference_dx::OffsetArray{T} = memory_gradient.reference_dx
    reference_dy::OffsetArray{T} = memory_gradient.reference_dy

    # Calculate derivate for x
    filter2davx!(reference_dx, image, kernel_dx, mode)

    # Calculate derivate for y
    filter2davx!(reference_dy, image, kernel_dy, mode)

    memory_gradient
end

@inline function sum_edges_angles(memory_gradient::MemoryGradient{T}, mode::Mode) where {T<:AbstractFloat}

    target_dx::Vector{T} = vec(memory_gradient.target_dx)
    target_dy::Vector{T} = vec(memory_gradient.target_dy)

    reference_dx::Vector{T} = vec(memory_gradient.reference_dx)
    reference_dy::Vector{T} = vec(memory_gradient.reference_dy)

    return get_gradient_oriented_sum(target_dx, target_dy, reference_dx, reference_dy, mode)

end

@inline function sum_exp_edges_angles(memory_gradient::MemoryGradient{T}, mode::Mode) where {T<:AbstractFloat}

    target_dx::Vector{T} = vec(memory_gradient.target_dx)
    target_dy::Vector{T} = vec(memory_gradient.target_dy)

    reference_dx::Vector{T} = vec(memory_gradient.reference_dx)
    reference_dy::Vector{T} = vec(memory_gradient.reference_dy)

    return get_gradient_exp_oriented_sum(target_dx, target_dy, reference_dx, reference_dy, mode)

end

end # end module
