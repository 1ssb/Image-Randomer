#!/usr/bin/env python3 

# Code by 1ssb: https://github.com/1ssb/Image-Randomizer

import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print("All Libraries imported!")

if not torch.cuda.is_available():
    raise Exception('GPU is not available')
device = torch.device('cuda')

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        return image

def select_diverse_images(source_distribution, k, batch):
    n = len(source_distribution)
    best_js_divergence = float('inf')
    best_subset = None
    for b in range(batch):
        subset = torch.randperm(n)[:k]
        sample_distribution = source_distribution[subset]
        m = 0.5 * (sample_distribution + source_distribution)
        js_divergence = 0.5 * torch.sum(sample_distribution * torch.log(sample_distribution / m) + source_distribution * torch.log(source_distribution / m))
        if js_divergence < best_js_divergence:
            best_js_divergence = js_divergence
            best_subset = subset
    return best_subset

def copy_diverse_images(source_path, dest_path, k, eval, batch):
    dataset = ImageDataset(source_path)
    dataloader = DataLoader(dataset, batch_size=1)
    
    source_distribution = []
    image_paths = dataset.image_paths
    
    for image in tqdm(dataloader):
        image_array = image.flatten()
        source_distribution.append(image_array)
    
    source_distribution = torch.cat(source_distribution)
    
    best_images = []
    for i in tqdm(range(eval)):
        dt = 0.1
        gamma = 1.0
        kT = 1.0 + i*dt
        source_distribution += -gamma * source_distribution * dt + np.sqrt(2 * gamma * kT * dt) * torch.randn_like(source_distribution)

        sample_indices = torch.randperm(len(source_distribution))[:5*k]
        sample_distribution = source_distribution[sample_indices]
        m = 0.5 * (sample_distribution + source_distribution)
        js_distance = 0.5 * (torch.sum(sample_distribution * torch.log(sample_distribution / m)) + torch.sum(source_distribution * torch.log(source_distribution / m)))
        
        best_subset = select_diverse_images(sample_distribution, k, batch)
        best_images.extend(best_subset.tolist())
        
    for i in best_images:
        shutil.copy(image_paths[i], dest_path)

source_path = '/path/to/source/images'
dest_path = '/path/to/destination/folder'
k = 10
eval = 50
batch = 50
copy_diverse_images(source_path, dest_path, k, eval)
