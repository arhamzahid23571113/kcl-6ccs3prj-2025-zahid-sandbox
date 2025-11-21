# Fastprior multi-target DE summary (CIFAR-10 test set, 10,000 indices)

## Image-wise success
- Total images attacked (correctly classified): **9213**
- Images with ≥1 successful target: **1310**
- Image-wise attack success rate: **0.142**

## Target-wise success
- Total target attempts (images × non-true classes): **82917**
- Successful target attacks: **1482**
- Target-wise success rate: **0.018**
- Mean #successful targets per successful image: **1.131**

## Optimisation cost
- Mean generations over all images: **40.0**
- Mean generations over successful images: **40.0**
- Mean queries over all images: **70848.0**
- Mean queries over successful images: **70848.0**

> Note: `total_images` will be slightly below 10,000 because only
> images that are correctly classified by the baseline model are attacked.
