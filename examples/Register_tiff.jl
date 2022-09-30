using Images
using Metaheuristics

using ImageRegistration
using RegisterImages
using Auxiliary
using RegisterMetrics
using RegisterRandom

## ========================================= Load images ================================================

file1 = "datasets/Hyperspectral/Lake/mono_lake-band1.tif"
file2 = "datasets/Hyperspectral/Lake/mono_lake-band4.tif"


image_target    = load(file1)
image_reference = load(file2)

# Convert to gray scale
gray_target     = Gray.(image_target)
gray_reference  = Gray.(image_reference)

mosaicview(gray_target, gray_reference; ncol = 2)

## ======================================== Create affine transformation ===============================================

defValue = NaN

# Create a bound search
bounds = get_vector_of_bounds(deg2rad(80.0),0.7,200.0)

# Get affine transforation 
affine_params = get_random_vector_from_interval(bounds)

# Apply the affine transformation
reference   = Float64.(gray_reference);
target      = warp(Float64.(gray_target), affine_params, defValue, PrivateThreads())

mosaicview(Gray.(target), gray_reference; ncol = 2)

## ================================================= Optimize ==========================================================
bins        = 16
options     = Options(f_calls_limit = 8000, f_tol = 1.0e-5)
x_optim     = affine_params;

#
mi_solution  = test_register_images(target,reference,bins,x_optim,bounds,options, MI(), OECA(), PrivateThreads());
hkp_solution = test_register_images(target,reference,bins,x_optim,bounds,options, HKP(),OECA(), PrivateThreads());

## =============================================== print solutions ===============================================
print(hkp_solution)
print(mi_solution)
