using Images
using Metaheuristics

using ImageRegistration
using RegisterImages
using RegisterMetrics
using RegisterRandom
using Utils
using Auxiliary


## ================================= Load image and Create Target and Reference=============================

file1 = "datasets/Multispectral/Pleyades3.jpg";

# Load image
image = load(file1);

# Convert to gray scale
float_image = Float64.(Gray.(image));

new_target  = image_spatially_varying(float_image, 3, 0.1, 80.0);
reference   = upgrade_intensity.(float_image);

field       = image_spatially_varying(ones(size(float_image)), 3, 0.3, 80.0);

mosaicview(Gray.(float_image), Gray.(field), Gray.(new_target), Gray.(reference); nrow = 1)

## ================================= Apply affine transform =========================================
# Create a bound search
bounds = get_vector_of_bounds(deg2rad(100.0),0.5,200.0)

# Get affine transforation 
x_optim = get_random_vector_from_interval(bounds)

# Apply the affine transformation
target  = warp(new_target, x_optim, NaN, PrivateThreads())

mosaicview(Gray.(target), Gray.(reference); ncol = 2)

## ================================= Register images== =========================================

bins    = 16
options = Options(f_calls_limit = 5000, x_tol=0.001, f_tol=0.001);

mi_gradient_solution    = test_register_gradient_images(target,reference,bins,x_optim,bounds,options, MI(), OECA(), PrivateThreads());
hkp_gradient_solution   = test_register_gradient_images(target,reference,bins,x_optim,bounds,options, HKP(),OECA(), PrivateThreads());

print(mi_gradient_solution)
print(hkp_gradient_solution)