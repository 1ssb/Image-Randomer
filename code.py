import os
import shutil
import numpy as np
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
        subset = torch.randperm(n)[:int(k)]
        sample_distribution = source_distribution[subset]
        
        # Pad with zeros
        max_len = source_distribution.shape[0]
        pad_len = max_len - sample_distribution.shape[0]
        sample_distribution = torch.cat([sample_distribution, torch.zeros((sample_distribution.shape[0], pad_len))], dim=1)
        pad_len = max_len - source_distribution.shape[1]
        source_distribution = torch.cat([source_distribution, torch.zeros((source_distribution.shape[0], pad_len))], dim=1)
        
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
        best_subset = select_diverse_images(source_distribution, k, batch)
        best_images.extend(best_subset.tolist())
        
    for i in best_images:
        shutil.copy(image_paths[i], dest_path)
'''       
def keep_diverse_images(path, P):
    if P >= 1 or P <= 0:
        raise ValueError("P must be less than one and greater than zero")
    
    # Load images
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    
    # Simpson's Diversity Index
    data = next(iter(dataloader))[0]
    n = len(data)
    sdi = []
    for i in tqdm(range(n), desc="Pruning to keep the most diverse set of images"):
        image = data[i]
        p = image / image.sum()
        sdi.append((1 - (p ** 2).sum()).item())
        
    sdi_sorted = sorted(range(n), key=lambda k: sdi[k], reverse=True)
    keep = int(n * P)
    for i in range(n):
        if i not in sdi_sorted[:keep]:
            os.remove(dataset.imgs[i][0])
'''
# I am trying something new. Don't bother right now.
source_path = '/src'
dest_path = '/dstr'
k = 100
evals = 50
batch = 50
# P = 0.5
# tot = k/P 
copy_diverse_images(source_path, dest_path, k, evals, batch)
# keep_diverse_images(dest_path, P)
