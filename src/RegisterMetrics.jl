module RegisterMetrics

using StaticArrays
using LinearAlgebra
using Printf

using RegisterImages
using Auxiliary
using RegisterMemory

import Base.show

export Metrics, affin_rmse, affin_mae

struct Metrics{T<:AbstractFloat}

    x_estimated::Vector{T}
    vnorm::T
    rmse::T 
    mae::T
    time::T
    fcalls::Int64
    iterations::Int64
    sim_name::String
    method_name::String
    gradient::String
    
    function Metrics(meta_results, x_optimim::Vector{T}, offset::SVector{2,T}, img_size::Tuple{Int64,Int64},
        simType::SIMType, optimizer::Optimizer; useGradient::Bool=false) where {T<:AbstractFloat}

        x_estimated = meta_results.best_sol.x

        # Euclidian norm beetween tru an real vector parameter
        vnorm = norm(x_estimated - x_optimim)

        # RMSE for the affin transformation
        rmse = affin_rmse(x_optimim, x_estimated, offset, img_size)

        # MAE for the affin transformation
        mae = affin_mae(x_optimim, x_estimated, offset, img_size)

        # Get time, function calls and iterations
        time = meta_results.overall_time
        fcalls = meta_results.f_calls
        iterations = meta_results.iteration

        sim_name    = name(simType)
        method_name = name(optimizer)

        gradient = useGradient ? "true" : "false"

        new{T}(x_estimated,vnorm,rmse,mae,time,fcalls,iterations, sim_name, method_name, gradient)
    end
end


function affin_rmse(x::Vector{T}, x_hat::Vector{T}, offset::SVector{2,T}, img_size::Tuple{Int64,Int64}) where {T<:AbstractFloat}

    # Get the real affin transformation
    M_true::SMatrix{2,2,T}, t_true::SVector{2,T} = get_affin_from_params(x, [offset[1], offset[2]])

    # Get the esimated affin transformation
    M_estimated::SMatrix{2,2,T}, t_estimated::SVector{2,T} = get_affin_from_params(x_hat, [offset[1], offset[2]])

    error::T = 0.0

    rows::Int64, cols::Int64 = img_size

    @simd for I in CartesianIndices((rows, cols))

        # True coordinates
        x_real, y_real = M_true * SVector(I.I) + t_true

        # Estimated coordinates
        x_est, y_est = M_estimated * SVector(I.I) + t_estimated

        # Get diferece
        error += (x_real - x_est) * (x_real - x_est) + (y_real - y_est) * (y_real - y_est)
    end

    return sqrt(error / (rows*cols))
end

function affin_mae(x::Vector{T}, x_hat::Vector{T}, offset::SVector{2,T}, img_size::Tuple{Int64,Int64}) where {T<:AbstractFloat}

    # Get the real affin transformation
    M_true::SMatrix{2,2,T}, t_true::SVector{2,T} = get_affin_from_params(x, [offset[1], offset[2]])

    # Get the esimated affin transformation
    M_estimated::SMatrix{2,2,T}, t_estimated::SVector{2,T} = get_affin_from_params(x_hat, [offset[1], offset[2]])

    rows::Int64, cols::Int64 = img_size
    error::T = 0.0
    @simd for I in CartesianIndices((rows, cols))

        # True coordinates
        x_real, y_real = M_true * SVector(I.I) + t_true

        # Estimated coordinates
        x_est, y_est = M_estimated * SVector(I.I) + t_estimated

        # Get diferece
        error += sqrt((x_real - x_est) * (x_real - x_est) + (y_real - y_est) * (y_real - y_est))
    end

    return error / (rows*cols)

end


function affin_rmse(points::Matrix{T}, H::AbstractMatrix{T}, epsilon::T) where{T<:AbstractFloat}
    
    count_points::Int64 = size(points,1)

    projected_points::Matrix{T} = zeros(T, count_points, 2)
    mask::Vector{Bool} = Vector{Bool}(undef, count_points)

    error::T        = 0.0

    @inbounds for i ∈ 1:count_points

        # True coordinates
        x_reference, y_reference, x_target, y_target,  = points[i, :]

        # Estimated coordinates
        x_est, y_est = H * SVector((x_reference, y_reference, 1))

        projected_points[i,1] = x_est
        projected_points[i,2] = y_est

        # Get diferece
        diff = (x_target - x_est)^2 + (y_est - y_target)^2

        if (sqrt(diff) < epsilon)
            mask[i] == true
        else
            mask[i] == false
        end

        error += diff

    end

    return sqrt(error/count_points), projected_points, mask

end

function affin_mae(points::Matrix{T}, H::AbstractMatrix{T}, epsilon::T) where{T<:AbstractFloat}
    
    count_points::Int64 = size(points,1)

    projected_points::Matrix{T} = zeros(T, count_points, 2)
    mask::Vector{Bool} = Vector{Bool}(undef, count_points)

    error::T        = 0.0

    @inbounds for i ∈ 1:count_points

        # True coordinates
        x_reference, y_reference, x_target, y_target,  = points[i, :]

        # Estimated coordinates
        x_est, y_est = H * SVector((x_reference, y_reference, 1))

        projected_points[i,1] = x_est
        projected_points[i,2] = y_est

        # Get diferece
        diff = (x_target - x_est)^2 + (y_est - y_target)^2

        if (sqrt(diff) < epsilon)
            mask[i] == true
        else
            mask[i] == false
        end

        error += sqrt(diff)

    end

    return error/count_points, projected_points, mask
end


function Base.show(io::IO, metrics::Metrics)

    println(io, "+=========== Solution ==========+")
    @printf(io,"%14s %s\n", "Similarity:", metrics.sim_name)
    @printf(io,"%14s %s\n", "Method:", metrics.method_name)
    @printf(io,"%14s %s\n", "Use gradient:", metrics.gradient)
    @printf(io,"%14s %.0f\n", "iterations:", metrics.iterations)
    @printf(io,"%14s %.0f\n", "f calls:", metrics.fcalls)
    @printf(io,"%14s %.4f s\n", "total time:", metrics.time)
    @printf(io,"%14s %.5f \n", "||x-x̂||:", metrics.vnorm)
    @printf(io,"%14s %.5f \n", "RMSE:", metrics.rmse)
    @printf(io,"%14s %.5f \n", "MAE:", metrics.mae)

    @printf(io,"%14s ", "minimizer:")
    show(io, metrics.x_estimated)
    println(io, "")

    println(io, "+=============================+\n")
end

end # end module
