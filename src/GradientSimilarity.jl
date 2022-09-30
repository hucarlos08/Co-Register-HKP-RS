module GradientSimilarity
    
using LoopVectorization

using RegisterMemory
using HistogramKP
using MutualInformation
using Auxiliary

export estimate_similarity_gradient_HKP!, estimate_similarity_gradient_MI!, cost_similarity_gradient

function estimate_similarity_gradient_HKP!(
    gradient_memory::MemoryGradient{T},
    params::Vector{T},
    simType::SIMType,
    mode::Mode,
) where {T<:AbstractFloat}


    # Warp image, estimate histogram and estimated sililarity
    sHKP::T = estimate_similarity_HKP!(gradient_memory.memory, params, simType, mode)

    # Estimated image derivatives
    get_gradient!(gradient_memory, mode)

    # Calculate gradient orientations cost
    # gradient_cost::T = sum_edges_angles(gradient_memory, mode)
    gradient_cost::T = sum_exp_edges_angles(gradient_memory, mode)

    return cost_similarity_gradient(sHKP, gradient_cost)

end

function estimate_similarity_gradient_MI!(
    gradient_memory::MemoryGradient{T},
    params::Vector{T},
    simType::SIMType,
    mode::Mode,
) where {T<:AbstractFloat}

    # Warp image, estimate histogram and estimated sililarity
    sMI::T = estimate_similarity_MI!(gradient_memory.memory, params, simType, mode)

    # Estimated image derivatives
    get_gradient!(gradient_memory, mode)

    # Calculate gradient orientations cost
    # gradient_cost::T = sum_edges_angles(gradient_memory, mode)
    gradient_cost::T = sum_exp_edges_angles(gradient_memory, mode)

    return cost_similarity_gradient(sMI, gradient_cost)
end


@inline function cost_similarity_gradient(similarity::T, gradient::T) where{T<:AbstractFloat}
    return similarity * gradient
end


end # end module