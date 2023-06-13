# Image Randomiser: A Method to Optimize Samples by Langevin Annealing using JS Divergence

# Description

This is an image sampling algorithm that samples a subset of images to optimize sample diversity by minimising the JS divergence between sample distributiuon and source distribution. We employ PyTorch to implement the image processing pipeline and can utilize graphical processing for acceleration if available. 

The core problem this code is solving is to select the most diverse images from a dataset using JS divergence loss function after annealing the distributions generated in the samples and the source. It adds noise to the images during the evaluation process using Langevin dynamics schedule to explore different solution space regions and avoid getting trapped in local minima.


Cite as: Bhattacharjee, S. S. (2023). Image Randomiser: A Method to Optimize Samples Using Downsampling and Langevin Annealing with JS Divergence, GitHub, retrieved from https://1ssb.github.io/Image-Randomiser/.
