#!/usr/bin/env python3
# Code by 1ssb
import os
import shutil
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    if not torch.cuda.is_available():
        raise Exception('GPU is not available')
    device = torch.device('cuda')
    print("All Libraries Imported, GPU in use!")

    class ImageDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = transforms.Resize((32, 32))(image)
            image = transforms.ToTensor()(image)
            return image.to(device)

    def select_diverse_images(source_distribution, k, batch):
        n = len(source_distribution)
        best_js_divergence = float('inf')
        best_subset = None
        for b in tqdm(range(batch), desc='Selecting diverse images'):
            subset = torch.randperm(n)[:int(k)]
            sample_distribution = source_distribution[subset]
            num_batches = n // k
            js_divergence = 0
            for i in range(num_batches):
                start = i * k
                end = start + k
                m = 0.5 * (sample_distribution + source_distribution[start:end])
                js_divergence += 0.5 * torch.sum(sample_distribution * torch.log(sample_distribution / m) + source_distribution[start:end] * torch.log(source_distribution[start:end] / m))
            js_divergence /= num_batches
            if js_divergence < best_js_divergence:
                best_js_divergence = js_divergence
                best_subset = subset
        if best_subset is None:
            return torch.randperm(n)[:int(k)]
        else:
            return best_subset

    def copy_diverse_images(source_path, dest_path, k, evals, batch):
        dataset = ImageDataset(source_path)
        dataloader = DataLoader(dataset, batch_size=16, num_workers=4)
        source_distribution = []
        image_paths = dataset.image_paths
        for image in tqdm(dataloader, desc='Loading images'):
            image_array = image.flatten()
            source_distribution.append(image_array)
        source_distribution = torch.cat(source_distribution).to(device)
        best_images = []
        for i in tqdm(range(evals), desc='Evaluating'):
            dt = 0.1
            gamma = 1.0
            kT = 1.0 + i*dt
            source_distribution += -gamma * source_distribution * dt + np.sqrt(2 * gamma * kT * dt) * torch.randn_like(source_distribution)
            best_subset = select_diverse_images(source_distribution, k, batch * 4)
            best_images.extend(best_subset.tolist())
        for i in tqdm(best_images, desc='Copying images'):
            shutil.copy(image_paths[i], dest_path)

    source_path = '/src'
    dest_path = '/dst'
    k = 100
    evals = 50
    batch = 50

    copy_diverse_images(source_path, dest_path
