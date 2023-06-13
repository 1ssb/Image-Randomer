#!/usr/bin/env python3
# code by 1ssb: https://github.com/1ssb/Image-Randomizer

import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print("All Libraries imported!")
# Check for GPU availibility
def process_images(source_path, destination_path, sample_size):
    if not torch.cuda.is_available():
        raise Exception('GPU is not available')
    device = torch.device('cuda')
# Load data into a virtual filesystem for feeding to the ImageLoader
    class ImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        # Read all the images in the path
        def __len__(self):
            return len(self.image_paths)
        # Transform iamge paths after reading
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
# Initialise the JS Divergence
    def loss_fn(sample_distribution, source_distribution):
        m = 0.5 * (sample_distribution + source_distribution)
        js_distance = 0.5 * (torch.sum(sample_distribution * torch.log(sample_distribution / m)) + torch.sum(source_distribution * torch.log(source_distribution / m)))
        return js_distance
# Add some noise to make the distibution a bit more robust when statistically computed
    def add_noise(image, level):
        noise = torch.randn_like(image) * level
        image = image + noise
        return image
# Transform the virtual image set to speed up calculations by downsampling
    transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    dataset = ImageDataset(source_path, transform=transform)
# Make sure a basic statistical parameterisation is maintained: You may change the value of 3, its just a basic heuristic.
    if len(dataset) < 3 * sample_size:
        raise Exception('The number of images in the source directory must be at least 3 times larger than the sample size')
# Prepare the list for saving the optimal image metadata (name)
    dataloader = DataLoader(dataset, batch_size=1)
    os.makedirs(destination_path, exist_ok=True)
# Initialize a list to store the optimal image names
optimal_image_names = []

# Iterate over images in the dataloader
for i, image in enumerate(tqdm(dataloader)):
    # Move the image to the device (e.g., GPU)
    image = image.to(device)
    # Flatten the image to create a 1D tensor representing the source distribution
    source_distribution = image.flatten()
    # Create a sample distribution by generating random noise with the same shape as the image
    sample_distribution = torch.randn_like(image).to(device)
    # Enable gradient computation for the sample distribution
    sample_distribution.requires_grad_(True)
    # Create an optimizer using the AdamW algorithm with the sample distribution as its parameter
    optimizer = torch.optim.AdamW([sample_distribution], lr=0.1)

    # Set the initial temperature and friction coefficient
    kT = 1.0
    gamma = 0.1

    # Perform 50 iterations of optimization to ensure convergence
    for j in range(50):
        # Compute the noise level based on the current temperature
        level = np.sqrt(2 * kT / gamma)
        # Generate a noisy version of the image by calling the add_noise function
        noisy_image = add_noise(image, level)
        # Update the sample distribution's data attribute with the flattened noisy image
        sample_distribution.data = noisy_image.flatten()
        # Compute the loss as the JS divergence between the sample and source distributions
        loss = loss_fn(sample_distribution, source_distribution)
        # Compute gradients by calling the backward method on the loss tensor
        loss.backward()
        # Update the sample distribution's parameters by calling the optimizer's step method
        optimizer.step()
        # Reset gradients to zero by calling the optimizer's zero_grad method
        optimizer.zero_grad()
        # Update the temperature using an annealing schedule: this is a control parameter.
        kT *= 0.95

    # Append the name of the optimal image to a list of optimal image names using its index in the dataset
    optimal_image_names.append(dataset.image_paths[i])

# Release memory used by tensors
    del image
    del source_distribution
    del sample_distribution
    torch.cuda.empty_cache()

# Making sure that the images which are copied are by index and the virtual images are killed after the iterations.
    for optimal_image_name in optimal_image_names[:sample_size]:
        original_image_name = os.path.basename(optimal_image_name)
        os.rename(optimal_image_name, os.path.join(destination_path, original_image_name))
    # Placeholders and call example: change this part of the code.
source_path = 'source'
destination_path = 'dest'
sample_size = #No
process_images(source_path, destination_path, sample_size)
