module RegisterRandom

export  get_value_from_interval,
        get_random_vector_from_interval,
        get_vector_of_bounds,
        get_random_affine,
        get_random_perspective


@inline function get_random_value_from_interval(bounds::Tuple{T, T}) where {T<:AbstractFloat}
    a::T = bounds[1]
    b::T = bounds[2]
    return ((b-a) * rand(T)) + a;
end


function get_random_vector_from_interval(bounds::Vector{Tuple{T, T}}) where {T<:AbstractFloat}

    params_count::Int64, = size(bounds)

    result::Vector{T} = zeros(T, params_count)
    for (index, bound) in enumerate(bounds)
        result[index] = get_random_value_from_interval(bound)
    end
    return result
end


@inline function get_vector_of_bounds(θ::T, λ::T, δ::T) where{T<:AbstractFloat}
    
    start::T = 1.0
    @assert(0.0 <= λ < 1.0)
    return [
                minmax(-θ, θ),                  # θ parameter
                minmax(start - λ, start + λ),   # λₓ
                minmax(start - λ, start + λ),   # λ𝚈
                minmax(-δ, δ),                  # dx
                minmax(-δ, δ),                  # dy
            ]
end


@inline function get_vector_of_bounds(θ::T, λ::T, δ, s::T) where{T<:AbstractFloat}
    
    start::T = 1.0
    @assert(0.0 <= λ < 1.0)
    return [
                minmax(-θ, θ),                  # θ parameter
                minmax(start - λ, start + λ),   # λₓ
                minmax(start - λ, start + λ),   # λ𝚈
                minmax(-δ, δ),                  # dx
                minmax(-δ, δ),                  # dy
                minmax(-s, s),                  # shear x
                minmax(-s, s)                   # shear y
            ]
end

function get_random_affine(θ::T, λ::T, δ::T) where{T<:AbstractFloat}
    
    bounds::Vector{Tuple{T,T}} = get_vector_of_bounds(θ, λ, δ)

    return get_random_vector_from_interval(bounds)
end


function get_random_perspective(θ::T, λ::T, δ::T, s::T) where{T<:AbstractFloat}
    
    bounds::Vector{Tuple{T,T}} = get_vector_of_bounds(θ, λ, δ, s)

    return get_random_vector_from_interval(bounds)
end


end