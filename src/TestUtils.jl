module TestUtils

using Metaheuristics
using LinearAlgebra

using AffineTransformation
using HistogramKP
using MutualInformation

export  testHKP,
        testNMI

## Function test HKP
function testHKP(target::Matrix{T},
    reference::Matrix{T},
    samples::Int64,
    transf_bounds::Array{Tuple{T,T},1},
    transformations::Matrix{T},
    idSim::Int64,
    nbins::Int64,
    defValue::Float64=NaN,
    f_calls::Int64=4000,
    tolerance::T=1.0e-3,
    methodName::String="ECA") where {T<:AbstractFloat}

    results = zeros(samples, 4)

    opt_bounds  = reshape(collect(Iterators.flatten(transf_bounds)),(2,5))

    # Image size
    w, h    = size(reference)

    for i=1:samples

        # Create the random image
        target_i = affineTransform(transformations[:, i], target, defValue)

        # Create the memory for avoid re-allocations
        hkp   = HKPMemory(reference, target_i, nbins, idSim, defValue)

        # Create the cost function
        f_HKP(x) = hkp(x,1)

        # First evaluation 
        f_HKP(transformations[:,i])

        # Create the optimization mutual parameters
        options     = Options(f_calls_limit = f_calls, f_tol = tolerance)
        information = Information(f_optimum = f_HKP(transformations[:,i]))

        method = Metaheuristics.Algorithm

        if(methodName == "ECA")
            method = ECA(information = information, options = options)

        elseif (methodName == "DE")
            method = DE(information = information, options = options)

        else
            throw("Not defined optimization method")
        end

        # Get results
        result  = Metaheuristics.optimize(f_HKP, opt_bounds, method)
        vnorm   = norm(minimizer(result) - transformations[:, i])
        verror  = affineError(minimizer(result), transformations[:,i], w, h)
        time    = result.overall_time
        fcalls  = result.f_calls

        # assign results
        results[i,1]  = vnorm
        results[i,2]  = verror
        results[i,3]  = time
        results[i,4]  = fcalls

    end

    return results
end

## Function TEST NMI
function testNMI(target::Matrix{T},
    reference::Matrix{T},
    samples::Int64,
    transf_bounds::Array{Tuple{T,T},1},
    transformations::Matrix{T},
    idSim::Int64,
    nbins::Int64,
    defValue::T=NaN,
    f_calls::Int64=6000,
    tolerance::T=1.0e-3,
    methodName::String="ECA") where {T<:AbstractFloat}

    results = zeros(samples, 4)

    opt_bounds  = reshape(collect(Iterators.flatten(transf_bounds)),(2,5))

    # Image size
    w, h    = size(reference)

    for i=1:samples

        # Create the random image
        target_i = affineTransform(transformations[:, i], target, defValue)

        # Create the memory for avoid re-allocations
        mi   = MIMemory(reference, target_i, nbins, idSim, defValue)

        # Create the cost function
        f_NMI(x) = mi(x,1)

        # First evaluation
        f_NMI(transformations[:,i])

        # Create the optimization mutual parameters
        options     = Options(f_calls_limit = f_calls, f_tol = tolerance)
        information = Information(f_optimum = f_NMI(transformations[:,i]))

        # Create the instances for the different algorithms
        method = Metaheuristics.Algorithm

        if(methodName == "ECA")
            method = ECA(information = information, options = options)

        elseif (methodName == "DE")
            method = DE(information = information, options = options)

        else
            throw("Not defined optimization method")
        end

        # Get results
        result  = Metaheuristics.optimize(f_NMI, opt_bounds, method)
        vnorm   = norm(minimizer(result) - transformations[:, i])
        verror  = affineError(minimizer(result), transformations[:,i], w, h)
        time    = result.overall_time
        fcalls  = result.f_calls

        # assign results
        results[i,1]  = vnorm
        results[i,2]  = verror
        results[i,3]  = time
        results[i,4]  = fcalls

    end

    return results
end


end # END MODULE