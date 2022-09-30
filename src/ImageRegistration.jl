module ImageRegistration

using Metaheuristics
using Images

using Auxiliary
using RegisterMemory
using KernelPredictability2
using HistogramKP
using MutualInformation
using RegisterMetrics
using GradientSimilarity

export test_register_images, test_register_gradient_images, estimate_similarity!, register_images, register_gradient_images


@inline function estimate_similarity(memory::Memory{T}, x::Vector{T}, ::KPAdd, mode::Mode) where {T<:AbstractFloat}
    estimate_similarity_KP(memory, x, KPAdd(), mode)
end

@inline function estimate_similarity(
    memory::Memory{T},
    x::Vector{T},
    ::KPNormalized,
    mode::Mode,
) where {T<:AbstractFloat}
    estimate_similarity_KP(memory, x, KPNormalized(), mode)
end

@inline function estimate_similarity!(memory::Memory{T}, x::Vector{T}, ::HKPAdd, mode::Mode) where {T<:AbstractFloat}
    estimate_similarity_HKP!(memory, x, HKPAdd(), mode)
end

@inline function estimate_similarity!(memory::Memory{T}, x::Vector{T}, ::HKP, mode::Mode) where {T<:AbstractFloat}
    estimate_similarity_HKP!(memory, x, HKP(), mode)
end

@inline function estimate_similarity!(memory::Memory{T}, x::Vector{T}, ::MI, mode::Mode) where {T<:AbstractFloat}
    estimate_similarity_MI!(memory, x, MI(), mode)
end

@inline function estimate_similarity!(
    memory::MemoryGradient{T},
    x::Vector{T},
    ::HKP,
    mode::Mode,
) where {T<:AbstractFloat}
    return estimate_similarity_gradient_HKP!(memory, x, HKP(), mode)
end

@inline function estimate_similarity!(
    memory::MemoryGradient{T},
    x::Vector{T},
    ::HKPAdd,
    mode::Mode,
) where {T<:AbstractFloat}
    return estimate_similarity_gradient_HKP!(memory, x, HKPAdd(), mode)
end

@inline function estimate_similarity!(
    memory::MemoryGradient{T},
    x::Vector{T},
    ::MI,
    mode::Mode,
) where {T<:AbstractFloat}
    return estimate_similarity_gradient_MI!(memory, x, MI(), mode)
end


function test_register_images(
    target::Matrix{T},
    reference::Matrix{T},
    nbins::Int64,
    optimum::Vector{T},
    bounds::Vector{Tuple{T,T}},
    options::Metaheuristics.Options,
    typeSIM::SIMType,
    optimizer::Optimizer,
    mode::Mode,
) where {T<:AbstractFloat}

    # Create the memory
    memory::Memory{T} = Memory(target, reference, nbins, convert(T, NaN))

    # Create function
    f(x) = estimate_similarity!(memory, x, typeSIM, mode)

    sol = get_solution(f, optimum, bounds, options, optimizer)

    # Estimate metrics
    return Metrics(sol, optimum, center(target), size(target), typeSIM, optimizer)
end


function test_register_gradient_images(
    target::Matrix{T},
    reference::Matrix{T},
    nbins::Int64,
    optimum::Vector{T},
    bounds::Vector{Tuple{T,T}},
    options::Metaheuristics.Options,
    typeSIM::SIMType,
    optimizer::Optimizer,
    mode::Mode,
) where {T<:AbstractFloat}

    # Create the memory
    memory::MemoryGradient{T} = MemoryGradient(target, reference, nbins, convert(T, NaN), mode)

    # Create function
    f(x) = estimate_similarity!(memory, x, typeSIM, mode)

    sol = get_solution(f, optimum, bounds, options, optimizer)

    # Estimate metrics
    return Metrics(sol, optimum, center(target), size(target), typeSIM, optimizer, useGradient = true)
end



function register_images(
    target::Matrix{T},
    reference::Matrix{T},
    nbins::Int64,
    bounds::Vector{Tuple{T,T}},
    fcalls::Int64,
    typeSIM::SIMType,
    optimizer::Optimizer,
    mode::Mode,
) where {T<:AbstractFloat}

    # Create the memory
    memory::Memory{T} = Memory(target, reference, nbins, convert(T, NaN))

    # Create options
    options = Options(f_calls_limit = fcalls);

    # Create function
    f(x) = estimate_similarity!(memory, x, typeSIM, mode)

    sol = get_solution(f, bounds, options, optimizer)

    return sol
end

function register_gradient_images(
    target::Matrix{T},
    reference::Matrix{T},
    nbins::Int64,
    bounds::Vector{Tuple{T,T}},
    fcalls::Int64,
    typeSIM::SIMType,
    optimizer::Optimizer,
    mode::Mode,
) where {T<:AbstractFloat}

    # Create the memory
    memory::MemoryGradient{T} = MemoryGradient(target, reference, nbins, convert(T, NaN), mode)

    # Create options
    options = Options(f_calls_limit = fcalls);

    # Create function
    f(x) = estimate_similarity!(memory, x, typeSIM, mode)

    sol = get_solution(f, bounds, options, optimizer)

    return sol
end


@inline function get_solution(
    f::Function,
    optimum::Vector{T},
    bounds::Vector{Tuple{T,T}},
    options::Metaheuristics.Options,
    optimizer::Optimizer,
) where {T<:AbstractFloat}

    # Create bounds
    meta_bounds::Matrix{T} = convert_vector_tuple_to_matrix(bounds)

    # Get information
    information = Information(x_optimum = optimum, f_optimum = f(optimum))

    # Optimize
    return optimize(f, options, information, meta_bounds, optimizer)
end


@inline function get_solution(
    f::Function,
    bounds::Vector{Tuple{T,T}},
    options::Metaheuristics.Options,
    optimizer::Optimizer,
) where {T<:AbstractFloat}

    # Create bounds
    meta_bounds::Matrix{T} = convert_vector_tuple_to_matrix(bounds)

    # Optimize
    return optimize(f, options, meta_bounds, optimizer)
end


function optimize(f::Function, _options, _bounds, ::OECA)

    algorithm   = ECA(options = _options)
    result      = Metaheuristics.optimize(f, _bounds, algorithm)

    return result
end

function optimize(f::Function, _options, _bounds, ::ODE)

    algorithm   = DE(options = _options)
    result      = Metaheuristics.optimize(f, _bounds, algorithm)

    return result
end

function optimize(f::Function, _options, _information, _bounds, ::OECA)

    algorithm = ECA(information = _information, options = _options)
    result = Metaheuristics.optimize(f, _bounds, algorithm)

    return result
end

function optimize(f::Function, _options, _information, _bounds, ::ODE)

    algorithm = DE(information = _information, options = _options)
    result = Metaheuristics.optimize(f, _bounds, algorithm)

    return result
end



end # end module