module Auxiliary

using StaticArrays

export findMinMax,
    minmax3,
    getbin,
    get_range,
    convert_vector_tuple_to_matrix,
    Mode,
    PrivateThreads,
    SIMD,
    NoParallelization,
    SIMType,
    KPAdd,
    KPNormalized,
    HKPAdd,
    HKP,
    MI,
    Optimizer,
    OECA,
    ODE,
    name


# "A trait for the ways the bin search and bin update steps can be parallelized."
abstract type Mode end

# "No threading nor vectorization."
struct NoParallelization <: Mode end
struct PrivateThreads <: Mode end
struct SIMD <: Mode end

# "A trait for the ways the similarit can be estimated."
abstract type SIMType end
struct KPAdd <: SIMType end
struct KPNormalized <: SIMType end
struct HKPAdd <: SIMType end
struct HKP <: SIMType end
struct MI <: SIMType end

#"A trait for the optimization methods."
abstract type Optimizer end

struct OECA <: Optimizer end
struct ODE <: Optimizer end

@inline function name(::KPAdd)
    return ("KPADD")
end
@inline function name(::KPNormalized)
    return ("KPNormalized")
end

@inline function name(::MI)
    return ("MI")
end

@inline function name(::HKPAdd)
    return ("HKPADD")
end
@inline function name(::HKP)
    return ("HKP")
end

@inline function name(::OECA)
    return ("ECA")
end
@inline function name(::ODE)
    return ("DE")
end



@inline function minmax3(v::T, vmin::T, vmax::T) where {T<:AbstractFloat}

    return isnan(v) ? (vmin, vmax) : (min(v, vmin), max(v, vmax))
end

function findMinMax(a::Vector{T}) where {T<:AbstractFloat}

    #Find minim and maximum
    minv::T = convert(T, Inf)
    maxv::T = convert(T, -Inf)

    @inbounds for value in a
        minv, maxv = minmax3(value, minv, maxv)
    end

    return minv, maxv
end

function findMinMax(x::Vector{T}, y::Vector{T}) where {T<:AbstractFloat}

    # Find minimum and maximun for every input
    min_x::T = convert(T, Inf)
    max_x::T = convert(T, -Inf)

    min_y::T = convert(T, Inf)
    max_y::T = convert(T, -Inf)

    @inbounds for i in eachindex(x)
        min_x, max_x = minmax3(x[i], min_x, max_x)
        min_y, max_y = minmax3(y[i], min_y, max_y)

    end

    return (min_x, max_x), (min_y, max_y)
end

@inline function getbin(v::T, min::T, norm::T, bins::Int) where {T<:AbstractFloat}
    return isnan(v) ? bins + 1 : clamp(trunc(Int, ((v - min) * norm) + 1), 1, bins)
end



@inline function get_range(n::Int64)

    tid = Threads.threadid()
    nt = Threads.nthreads()
    d, r = divrem(n, nt)

    from = (tid - 1) * d + min(r, tid - 1) + 1
    to = from + d - 1 + (tid ≤ r ? 1 : 0)
    from:to
end

function convert_vector_tuple_to_matrix(input::Vector{Tuple{T,T}}) where {T<:AbstractFloat}

    D::Int64, = size(input)
    return reshape(collect(Iterators.flatten(input)), (2, D))
end

end # end module
