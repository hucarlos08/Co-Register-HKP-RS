using Images
using ImageDraw
using Metaheuristics
using Interpolations


push!(LOAD_PATH, "src/")
using ImageRegistration
using RegisterImages
using Auxiliary
using RegisterMetrics
using RegisterRandom
using RegisterMemory
using GradientSimilarity
using Utils
using RegisterMetrics

using DelimitedFiles
using StaticArrays
using ImageDraw
using CSV
using DataFrames
using NPZ

## ========================================= Load DF infos ================================================

df      = CSV.read("datasets/GroundCP/Datasets.csv", DataFrame);

dataset = eachcol(df[df.Name .== "Lidar-Optical", :]);

path            = dataset.Path[1];
file_target     = path * dataset.Target[1];
file_reference  = path * dataset.Reference[1];
file_cp         = path * dataset.CP_csv[1];
file_aff_hopc   = path * dataset.Affine_HOPC[1];
output_path     = path * dataset.Output_Path[1];

## ========================================= Load images ================================================

image_target = RGB.(load(file_target));
image_reference = RGB.(load(file_reference));

gray_target, gray_reference = same_size_gray(image_target, image_reference);

mosaicview(gray_target, gray_reference; ncol = 2)

## ========================================= Plot Control Points ================================================
cp = readdlm(file_cp, ',', Float64, '\n')
cp = cp

cp_target =    @view cp[:,1:2];
cp_reference = @view cp[:,3:4];

image_cp_target       = draw_corners(image_target, cp_target,     RGB(1,0,0), 4);
image_cp_reference    = draw_corners(image_reference, cp_reference,  RGB(1,0,0), 4);

output_image_cp_target      = output_path * "target_cp.png";
output_image_cp_reference   = output_path * "reference_cp.png";

Images.save(output_image_cp_target, image_cp_target);
Images.save(output_image_cp_reference, image_cp_reference);

mosaicview(image_cp_target, image_cp_reference; ncol = 2)

## ========================================= Register images ================================================
target      = Float64.(gray_target);
reference   = Float64.(gray_reference);

# Set bins
nbins = 16;
# Create a bound search
bounds = get_vector_of_bounds(deg2rad(5.0), 0.2, 10.0);

# Maximum call function
fcalls = 10000;

#solution = register_images(target, reference, nbins, bounds, fcalls, HKP(), OECA(), PrivateThreads())
solution = register_gradient_images(target, reference, nbins, bounds, fcalls, HKP(), OECA(), PrivateThreads());

reference_result = warp(reference, minimizer(solution), NaN, PrivateThreads());

zero_image = zeros(size(reference));

## ================================================= VIEW HKP =================================================

# Estimate difference
diff1 = adjust_histogram((target-reference), LinearStretching());
diff2 = adjust_histogram((target-reference_result), LinearStretching());

diff1_name = output_path * "diff1.png";
diff2_name = output_path * "diff2.png";

Images.save(diff1_name, Gray.(diff1));
Images.save(diff2_name, Gray.(map(clamp01nan, diff2)));

mosaicview(Gray.(diff1), Gray.(diff2); ncol = 2)

## ================================================= MOSAIC VIEW =================================================
steps = (7,7)

reference_result_RGB = warp_multiple_channels(image_reference, minimizer(solution), NaN);
mosaic_init = create_mosaic_view(image_target, image_reference, steps)
mosaic_result= create_mosaic_view(image_target, reference_result_RGB, steps)

mosaic1_name = output_path * "mosaic1.png";
mosaic2_name = output_path * "mosaic2.png"

Images.save(mosaic1_name, mosaic_init);
Images.save(mosaic2_name, mosaic_result);