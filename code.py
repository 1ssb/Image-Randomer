#!/usr/bin/env python3

# Standalone code by 1ssb

# Image_Randomer


import os
import shutil
from PIL import Image
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from scipy.stats import ks_2samp

# Check if the required packages are installed and install them if necessary
try:
    from scipy.spatial.distance import cdist, jensenshannon
except ImportError:
    import subprocess
    subprocess.check_call(["python3", '-m', 'pip', 'install', 'scipy'])
    from scipy.spatial.distance import cdist, jensenshannon

try:
    from pytorch_fid import fid_score
except ImportError:
    import subprocess
    subprocess.check_call(["python3", '-m', 'pip', 'install', 'pytorch-fid'])
    from pytorch_fid import fid_score

print("All libraries successfully imported")

# Placeholder variables
src_path = "/path/to/images"
dst_path = "/path/to/sample"
target_number = 200
importance_values = [1, 1, 1, 1, 1]

#    Importance value rules are:
#    w1 = mahalanobis_distance_importance   
#    w2 = js_divergence_importance
#    w3 = fid_score_value_importance
#    w4 = kl_divergence_importance
#    w5 = ks_distance_importance

def create_image_distribution(src_path, target_number, dst_path, importance_values):
    # Read images from the src_path
    images = []
    filenames = []
    for filename in os.listdir(src_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(src_path, filename))
            images.append(np.array(img))
            filenames.append(filename)
    
    # Check if the source number is at least 3 times the target number
    if len(images) < 3 * target_number:
        raise ValueError("The source number must be at least 3 times the target number")
    
    # Convert the list of images to a numpy array and move it to the GPU
    images = torch.tensor(images).to('cuda')
    
    # Calculate the mean and standard deviation of the pixel values
    mean = torch.mean(images, axis=(0,1,2))
    std = torch.std(images, axis=(0,1,2))
    
    # Create a normal distribution using the calculated mean and standard deviation
    distribution = torch.normal(mean=mean, std=std, size=images.shape)
    
    # Initialize the minimum distance and best sample
    min_distance = float("inf")
    best_sample = None
    
    # Define a basic CNN model for downsampling the images (using float32 to reduce memory usage)
    class DownsampleCNN(nn.Module):
        def __init__(self):
            super(DownsampleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        def forward(self, x):
            x = x.float()
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = self.conv3(x)
            x = nn.functional.relu(x)
            return x
    
    # Move the CNN model to the GPU and set it to evaluation mode (to disable dropout and batch normalization)
    cnn_model = DownsampleCNN().to('cuda')
    cnn_model.eval()
    
    # Iterate len(images) times with a progress bar over the entire operation
    with tqdm(total=len(images)) as pbar:
        for i in range(len(images)):
            # Randomly select target_number images
            sample_indices = torch.randperm(len(images))[:target_number]
            sample_images = images[sample_indices]
            
            # Downsample the images using the CNN model (using float32 to reduce memory usage)
            sample_images_downsampled = cnn_model(sample_images.permute(0,3,1,2)).permute(0,2,3,1)
            
            # Simulate Langevin dynamics to add noise to the downsampled images before calculating statistical distances
            dt = 0.1  # Time step size for Langevin dynamics simulation
            gamma = 1.0  # Friction coefficient for Langevin dynamics simulation
            kT = 1.0  # Temperature for Langevin dynamics simulation (kT is Boltzmann constant times temperature)
            
            sample_images_downsampled += -gamma * sample_images_downsampled * dt + np.sqrt(2 * gamma * kT * dt) * torch.randn_like(sample_images_downsampled)
            
            # Calculate the mean and standard deviation of the sample pixel values (using float32 to reduce memory usage)
            sample_mean = torch.mean(sample_images_downsampled.float(), axis=(0,1,2))
            sample_std = torch.std(sample_images_downsampled.float(), axis=(0,1,2))
            
            # Create a normal distribution for the sample using the calculated mean and standard deviation (using float32 to reduce memory usage)
            sample_distribution = torch.normal(mean=sample_mean.float(), std=sample_std.float(), size=sample_images_downsampled.shape)
            
            # Calculate the Mahalanobis distance between the two distributions (using NumPy)
            mahalanobis_distance = cdist(distribution.cpu().numpy().reshape(1,-1), sample_distribution.cpu().numpy().reshape(1,-1), metric='mahalanobis')[0][0]
            
            # Calculate the Jensen-Shannon divergence between the two distributions (using SciPy)
            js_divergence = jensenshannon(distribution.cpu().numpy().flatten(), sample_distribution.cpu().numpy().flatten())
            
            # Calculate the FID score between the two distributions (using pytorch-fid)
            mu1, sigma1 = fid_score.calculate_activation_statistics(distribution.cpu().numpy().reshape(target_number,-1), model=None)
            mu2, sigma2 = fid_score.calculate_activation_statistics(sample_distribution.cpu().numpy().reshape(target_number,-1), model=None)
            fid_score_value = fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            
            # Calculate the KL divergence between the two distributions (using PyTorch)
            kl_divergence = torch.nn.functional.kl_div(torch.log_softmax(distribution.flatten(), dim=0), torch.softmax(sample_distribution.flatten(), dim=0))
            
            # Calculate the Kolmogorov-Smirnov distance between the two distributions (using SciPy)
            ks_distance = ks_2samp(distribution.cpu().numpy().flatten(), sample_distribution.cpu().numpy().flatten()).statistic
            
            # Assign placeholder values for the importance of the statistical parameters
            mahalanobis_distance_importance = importance_values[0]
            js_divergence_importance = importance_values[1]
            fid_score_value_importance = importance_values[2]
            kl_divergence_importance = importance_values[3]
            ks_distance_importance = importance_values[4]
            
            # Calculate the weights as the normalized importance values
            w1 = mahalanobis_distance_importance / sum(importance_values)
            w2 = js_divergence_importance / sum(importance_values)
            w3 = fid_score_value_importance / sum(importance_values)
            w4 = kl_divergence_importance / sum(importance_values)
            w5 = ks_distance_importance / sum(importance_values)
            
            # Calculate the total loss as a weighted sum of the individual loss values
            total_loss = w1 * mahalanobis_distance + w2 * js_divergence + w3 * fid_score_value + w4 * kl_divergence + w5 * ks_distance
            
            # Update the minimum distance and best sample if necessary
            if total_loss < min_distance:
                min_distance = total_loss
                best_sample = sample_indices
            
            # Update the progress bar
            pbar.update(1)
    
    # Copy the best sample images to the dst_path (if it doesn't already exist)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    for i in best_sample:
        shutil.copy(os.path.join(src_path, filenames[i]), os.path.join(dst_path, filenames[i]))
    
    return distribution

# Example usage
distribution = create_image_distribution(src_path, target_number, dst_path, importance_values)