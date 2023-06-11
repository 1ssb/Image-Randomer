# Image Randomiser: A Method to optimize samples using a downsampling and Langevin annealing using JS Divergence.

# Function is GPU optimized so as to make sure that the operations are well optimized.

# Code by 1ssb.

# Image Processing with Langevin Dynamics

This repository contains a Python script that uses Langevin dynamics to process a set of images and save a sample of the optimal images to a destination directory.

## Overview

The script loads a set of images from a source directory and downsamples them to speed up computation. It then adds noise to the images during the evaluation process using Langevin dynamics and remembers the names of the optimal images in the sample distribution. After the evaluation process has completed, it copies only those images from the source to the destination directory.

The code uses PyTorch to implement the image processing pipeline and can use multiple GPUs for acceleration if they are available.

## Jensen-Shannon Divergence

The script uses the Jensen-Shannon (JS) divergence as a measure of similarity between two probability distributions. The JS divergence is a symmetrized and smoothed version of the Kullback-Leibler (KL) divergence, which measures how different two probability distributions are.

The motivation behind using the JS divergence is that it is a bounded measure of similarity, meaning that its value always lies between 0 and 1. This makes it easier to interpret and compare the results. Additionally, unlike the KL divergence, the JS divergence is symmetric, meaning that it gives the same result regardless of the order in which the two distributions are compared.

## Usage

To use the script, you need to specify the source and destination directories and the sample size as arguments to the `process_images` function. The source directory should contain the images you want to process, and the destination directory is where the resulting images will be saved. The sample size determines how many of the optimal images will be saved to the destination directory.

CHECK OUT THE CODE AND MAIL ME IF SOMETHING MESSES UP at Subhransu.Bhattacharjee@anu.edu.au

#   References

1. Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th International Conference on Machine Learning (ICML-11) (pp. 681-688).
2. Li, C., Zhu, J., & Zhang, B. (2018). Learning energy-based models with exponential family Langevin dynamics. arXiv preprint arXiv:1811.12359.
3. Pascanu, R., & Bengio, Y. (2014). Revisiting natural gradient for deep networks. arXiv preprint arXiv:1301.3584.
4. Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.
