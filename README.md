# Image Randomiser: A Method to Optimize Samples by Downsampling and Langevin Annealing using JS Divergence.

Description: This Python script that processes a set of images to optimize samples using downsampling and Langevin annealing with JS divergence. We employ PyTorch to implement the image processing pipeline and can utilize multiple GPUs for acceleration if available. The core problem this code is solving is to select the most diverse images from a dataset using JS divergence method after annealing the distributions generated in the samples and the source. It adds noise to the images during the evaluation process using Langevin dynamics to enable the optimization algorithm to explore different solution space regions and avoid getting trapped in local minima.


Cite as: Bhattacharjee, S. S. (2023). Image Randomiser: A Method to Optimize Samples Using Downsampling and Langevin Annealing with JS Divergence, GitHub, retrieved from https://1ssb.github.io/Image-Randomiser/.
