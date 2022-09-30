module Tif

using ArchGDAL

export loadTif,
loadUnormedTiff,
normalize

function normalize(array::AbstractMatrix, a=0.0, b=1.0)

    xmin = Inf32
    xmax = -Inf32

    result = zeros(Float64, size(array))

    @inbounds for val in array
        xmin = min(xmin, val)
        xmax = max(xmax, val)
    end

    scale = (b-a)/(xmax - xmin)
    @simd for i in eachindex(array)
        result[i] = (array[i] - xmin)*scale + a
    end

    return result
end


function loadTif(filename::String)

    dataset = ArchGDAL.read(filename)

    number_rasters  = (ArchGDAL.nraster(dataset))
    width           = ArchGDAL.width(dataset)
    height          = ArchGDAL.height(dataset)

    m = Array{Float64}(undef, height, width, number_rasters)

    for i = 1:number_rasters

        band = ArchGDAL.getband(dataset,i)
        b    = normalize(ArchGDAL.read(band))

        m[:, :, i] = permutedims(b, (2,1))
    end

    return  m

end


function loadTif(filename::String, bands::Array{Int64})

    dataset = ArchGDAL.read(filename)

    width           = ArchGDAL.width(dataset)
    height          = ArchGDAL.height(dataset)

    m = Array{Float64}(undef, height, width, length(bands))

    for (i, b) in enumerate(bands)

        band    = ArchGDAL.getband(dataset,b)

        bnorm   = normalize(ArchGDAL.read(band))

        m[:, :, i] = permutedims(bnorm, (2,1))
    end

    return  m

end

function loadUnormedTiff(filename::String, bands::Array{Int64})

    dataset = ArchGDAL.read(filename)

    width           = ArchGDAL.width(dataset)
    height          = ArchGDAL.height(dataset)

    m = Array{Float64}(undef, height, width, length(bands))

    for (i, b) in enumerate(bands)

        band    = ArchGDAL.getband(dataset,b)

        bnorm   = ArchGDAL.read(band)

        m[:, :, i] = permutedims(bnorm, (2,1))
    end

    return  m
end

end
