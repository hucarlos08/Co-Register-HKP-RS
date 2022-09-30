module RegisterFilters

using LoopVectorization
using OffsetArrays
using Auxiliary


export filter2davx!,
    gradient_magnitude!,
    angle_weight,
    get_gradient_oriented_sum,
    get_gradient_oriented_sum!,
    get_gradient_exp_oriented_sum

function filter2davx!(
    out::AbstractMatrix{T},
    A::AbstractMatrix{T},
    kern::AbstractMatrix{T},
    ::NoParallelization,
) where {T<:AbstractFloat}

    for J in CartesianIndices(out)
        tmp = zero(eltype(out))
        for I ∈ CartesianIndices(kern)
            tmp += !isnan(A[I+J]) ? (A[I+J] * kern[I]) : (0.0)
        end
        out[J] = tmp
    end
    out
end


function filter2davx!(
    out::AbstractMatrix{T},
    A::AbstractMatrix{T},
    kern::AbstractMatrix{T},
    ::SIMD,
) where {T<:AbstractFloat}

    @turbo for J in CartesianIndices(out)
        tmp = zero(eltype(out))
        for I ∈ CartesianIndices(kern)
            @inbounds tmp += !isnan(A[I+J]) ? (A[I+J] * kern[I]) : (0.0)
        end
        @inbounds out[J] = tmp
    end
    out
end


function filter2davx!(
    out::AbstractMatrix{T},
    A::AbstractMatrix{T},
    kern::AbstractMatrix{T},
    ::PrivateThreads,
) where {T<:AbstractFloat}

    @tturbo for J in CartesianIndices(out)
        tmp = zero(eltype(out))
        for I ∈ CartesianIndices(kern)
            @inbounds tmp += !isnan(A[I+J]) ? (A[I+J] * kern[I]) : (0.0)
        end
        @inbounds out[J] = tmp
    end
    out
end

function gradient_magnitude!(
    out::OffsetArray{T},
    dx::OffsetArray{T},
    dy::OffsetArray{T},
    ::NoParallelization,
) where {T<:AbstractFloat}
    for I ∈ CartesianIndices(out)
        out[I] = sqrt((dx[I] * dx[I]) + (dy[I] * dy[I]))
    end
    out
end

function gradient_magnitude!(
    out::OffsetArray{T},
    dx::OffsetArray{T},
    dy::OffsetArray{T},
    ::SIMD,
) where {T<:AbstractFloat}
    @turbo for I ∈ CartesianIndices(out)
        @inbounds out[I] = sqrt((dx[I] * dx[I]) + (dy[I] * dy[I]))
    end
    out
end

function gradient_magnitude!(
    out::OffsetArray{T},
    dx::OffsetArray{T},
    dy::OffsetArray{T},
    ::PrivateThreads,
) where {T<:AbstractFloat}
    @tturbo for I ∈ CartesianIndices(out)
        @inbounds out[I] = sqrt((dx[I] * dx[I]) + (dy[I] * dy[I]))
    end
    out
end

@inline function angle_weight(angle::T) where {T<:AbstractFloat}

    return (cos(2.0 * angle) + 1.0) / 2.0
end

function get_gradient_oriented_sum(
    a_dx::Vector{T},
    a_dy::Vector{T},
    b_dx::Vector{T},
    b_dy::Vector{T},
    ::NoParallelization,
) where {T<:AbstractFloat}


    result::T = 0.0
    ϵ::T = 1.0e-8

    for I ∈ eachindex(a_dx)

        a_magnitude = sqrt((a_dx[I] * a_dx[I]) + (a_dy[I] * a_dy[I]))
        b_magnitude = sqrt((b_dx[I] * b_dx[I]) + (b_dy[I] * b_dy[I]))

        dot_a_b = (a_dx[I] * b_dx[I]) + (a_dy[I] * b_dy[I])
        angle_a_b = acos(dot_a_b / ((a_magnitude * b_magnitude) + ϵ))

        weight = angle_weight(angle_a_b)

        angle_a_b = dot_a_b / ((a_magnitude * b_magnitude) + ϵ)

        result += min(a_magnitude, b_magnitude) * weight

    end

    result / length(a_dx)

end

function get_gradient_oriented_sum(
    a_dx::Vector{T},
    a_dy::Vector{T},
    b_dx::Vector{T},
    b_dy::Vector{T},
    ::SIMD,
) where {T<:AbstractFloat}


    result::T = 0.0
    ϵ::T = 1.0e-8
    @turbo for I ∈ eachindex(a_dx)

        @inbounds a_magnitude = sqrt((a_dx[I] * a_dx[I]) + (a_dy[I] * a_dy[I]))
        @inbounds b_magnitude = sqrt((b_dx[I] * b_dx[I]) + (b_dy[I] * b_dy[I]))

        @inbounds dot_a_b = (a_dx[I] * b_dx[I]) + (a_dy[I] * b_dy[I])

        angle_a_b = acos(dot_a_b / ((a_magnitude * b_magnitude) + ϵ))
        weight = (cos(2.0 * angle_a_b) + 1.0) / 2.0

        result += min(a_magnitude, b_magnitude) * weight
    end

    result / length(a_dx)

end


function get_gradient_oriented_sum(
    a_dx::Vector{T},
    a_dy::Vector{T},
    b_dx::Vector{T},
    b_dy::Vector{T},
    ::PrivateThreads,
) where {T<:AbstractFloat}


    result::Vector{T} = zeros(T, Threads.nthreads())
    ϵ::T = 1.0e-8

    nthreads::Int64 = Threads.nthreads()
    k_inputs::Int64, = size(a_dx)

    Threads.@threads for thread_idx = 1:nthreads

        # Get memory reference
        partial_sum_idx::T = 0.0

        @turbo for i in get_range(k_inputs)

            @inbounds a_magnitude = sqrt((a_dx[i] * a_dx[i]) + (a_dy[i] * a_dy[i]))
            @inbounds b_magnitude = sqrt((b_dx[i] * b_dx[i]) + (b_dy[i] * b_dy[i]))

            @inbounds dot_a_b = (a_dx[i] * b_dx[i]) + (a_dy[i] * b_dy[i])

            angle_a_b = acos(dot_a_b / ((a_magnitude * b_magnitude) + ϵ))

            weight = (cos(2.0 * angle_a_b) + 1.0) / 2.0

            partial_sum_idx += min(a_magnitude, b_magnitude) * weight

        end

        result[thread_idx] = partial_sum_idx
    end

    sum(result)# / length(a_dx)

end



function get_gradient_exp_oriented_sum(
    a_dx::Vector{T},
    a_dy::Vector{T},
    b_dx::Vector{T},
    b_dy::Vector{T},
    ::PrivateThreads,
) where {T<:AbstractFloat}


    result::Vector{T} = zeros(T, Threads.nthreads())
    ϵ::T = 1.0e-8
    λ::T = 1.0

    nthreads::Int64 = Threads.nthreads()
    k_inputs::Int64, = size(a_dx)

    Threads.@threads for thread_idx = 1:nthreads

        # Get memory reference
        partial_sum_idx::T = 0.0

        @turbo for i in get_range(k_inputs)

            @inbounds a_magnitude = sqrt((a_dx[i] * a_dx[i]) + (a_dy[i] * a_dy[i]))
            @inbounds b_magnitude = sqrt((b_dx[i] * b_dx[i]) + (b_dy[i] * b_dy[i]))

            @inbounds dot_a_b = (a_dx[i] * b_dx[i]) + (a_dy[i] * b_dy[i])

            angle_a_b = dot_a_b / ((a_magnitude * b_magnitude) + ϵ)

            weight = 1.0 - exp(-λ * (angle_a_b * angle_a_b))

            partial_sum_idx += min(a_magnitude, b_magnitude) * weight

        end

        result[thread_idx] = partial_sum_idx
    end

    sum(result)

end


end # end module
