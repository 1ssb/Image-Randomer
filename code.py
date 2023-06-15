#!/usr/bin/env python3
# Code by 1ssb

# Import necessary libraries
import os
import shutil
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Check if required libraries are installed and install them if necessary
try:
    import PIL
    import torch
    import torchvision
    import tqdm
except ImportError:
    import subprocess
    subprocess.check_call(["python3", '-m', 'pip', 'install', 'Pillow', 'torch', 'torchvision', 'tqdm'])

def select_diverse_images(source_distribution, k, batch):
    """
    This function selects a subset of k images from the source_distribution that are most diverse.
    It does this by computing the Jensen-Shannon divergence between the sample_distribution and the source_distribution.
    The function returns the indices of the k most diverse images in the source_distribution.
    """
    # Get the number of images in the source_distribution
    n = len(source_distribution)
    
    # Initialize variables to keep track of the best subset of images and its Jensen-Shannon divergence
    best_js_divergence = float('inf')
    best_subset = None
    
    # Iterate over the specified number of batches
    for b in tqdm(range(batch), desc='Selecting diverse images'):
        # Select a random subset of k images from the source_distribution
        subset = torch.randperm(n)[:int(k)]
        sample_distribution = source_distribution[subset]
        
        # Compute the number of batches in the source_distribution
        num_batches = n // k
        
        # Initialize variable to keep track of the Jensen-Shannon divergence for this subset of images
        js_divergence = 0
        
        # Iterate over all batches in the source_distribution
        for i in range(num_batches):
            start = i * k
            end = start + k
            
            # Compute the average distribution between the sample_distribution and this batch of images from the source_distribution
            m = 0.5 * (sample_distribution + source_distribution[start:end])
            
            # Compute the Jensen-Shannon divergence between the sample_distribution and this batch of images from the source_distribution and add it to js_divergence
            js_divergence += 0.5 * torch.sum(sample_distribution * torch.log(sample_distribution / m) + source_distribution[start:end] * torch.log(source_distribution[start:end] / m))
        
        # Compute the average Jensen-Shannon divergence for this subset of images by dividing js_divergence by num_batches
        js_divergence /= num_batches
        
        # Check if this subset of images has a lower Jensen-Shannon divergence than the current best subset of images and update best_subset and best_js_divergence if necessary
        if js_divergence < best_js_divergence:
            best_js_divergence = js_divergence
            best_subset = subset
    
        # Check if a best_subset was found and return it if it was, otherwise return a random subset of k images from the source_distribution
        if best_subset is None:
            return torch.randperm(n)[:int(k)]
        else:
            return best_subset
        
def copy_diverse_images(source_path, dest_path, k, evals, batch):
   # Get a list of all image paths in the src directory
    image_paths = [os.path.join(src, f) for f in os.listdir(src) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
    
    # Initialize an empty list to store the source_distribution
    source_distribution = []
    
    # Iterate over all image paths in the src directory
    for image_path in tqdm(image_paths, desc='Loading images'):
        # Open the image and convert it to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Downsample the image to a size of (32, 32)
        image = transforms.Resize((32, 32))(image)
        
        # Convert the downsampled image to a tensor and flatten it
        image_array = transforms.ToTensor()(image).flatten()
        
        # Append the flattened image tensor to the source_distribution list
        source_distribution.append(image_array)
    
    # Stack all flattened image tensors in the source_distribution list into a single tensor
    source_distribution = torch.stack(source_distribution)
    
    # Initialize an empty list to store the indices of the best images
    best_images = []
    
    # Iterate over the specified number of evaluations
    for i in tqdm(range(evals), desc='Evaluating'):
        dt = 0.1
        gamma = 1.0
        kT = 1.0 + i*dt
        
        # Update the source_distribution using Langevin dynamics
        source_distribution += -gamma * source_distribution * dt + np.sqrt(2 * gamma * kT * dt) * torch.randn_like(source_distribution)
        
        # Call select_diverse_images to select the k most diverse images from the updated source_distribution
        best_subset = select_diverse_images(source_distribution, k, batch * 4)
        
        # Append the indices of the selected images to the best_images list
        best_images.extend(best_subset.tolist())
    
    # Iterate over all indices in the best_images list
    for i in tqdm(best_images, desc='Copying images'):
        # Copy the selected image from the src directory to the dst directory without modifying it
        shutil.copy(image_paths[i], dst)

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise Exception('GPU is not available')
    
    # Set device to cuda if GPU is available
    device = torch.device('cuda')
    
    print("All Libraries Imported, GPU in use!")
    
    # Set src and dst placeholders for source and destination paths
    src = '/src'
    dst = '/dst'
    
    # Assign parameters
    k = 2
    evals = 1
    batch = 1
    
    # Call copy_diverse_images to copy k most diverse images from src directory to dst directory
    copy_diverse_images(src, dst, k, evals, batch)
