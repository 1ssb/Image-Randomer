#!/usr/bin/env python3
# code by 1ssb: https://github.com/1ssb/Image-Randomizer

import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def process_images(source_path, destination_path, sample_size):
    if not torch.cuda.is_available():
        raise Exception('GPU is not available')
    device = torch.device('cuda')

    class ImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image

    def loss_fn(sample_distribution, source_distribution):
        m = 0.5 * (sample_distribution + source_distribution)
        js_distance = 0.5 * (torch.sum(sample_distribution * torch.log(sample_distribution / m)) + torch.sum(source_distribution * torch.log(source_distribution / m)))
        return js_distance

    def add_noise(image, level):
        noise = torch.randn_like(image) * level
        image = image + noise
        return image

    transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    dataset = ImageDataset(source_path, transform=transform)

    if len(dataset) < 3 * sample_size:
        raise Exception('The number of images in the source directory must be at least 3 times larger than the sample size')

    dataloader = DataLoader(dataset, batch_size=1)
    os.makedirs(destination_path, exist_ok=True)
    optimal_image_names = []

    for i, image in enumerate(tqdm(dataloader)):
        image = image.to(device)
        source_distribution = image.flatten()
        sample_distribution = torch.randn_like(image).to(device)
        sample_distribution.requires_grad_(True)
        optimizer = torch.optim.AdamW([sample_distribution], lr=0.1)

        for j in range(50):
            for level in [0.1, 0.01, 0.001]:
                noisy_image = add_noise(image, level)
                sample_distribution.data = noisy_image.flatten()
                loss = loss_fn(sample_distribution, source_distribution)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        optimal_image_names.append(dataset.image_paths[i])

    for optimal_image_name in optimal_image_names[:sample_size]:
        original_image_name = os.path.basename(optimal_image_name)
        os.rename(optimal_image_name, os.path.join(destination_path, original_image_name))

source_path = 'source'
destination_path = 'dest'
sample_size = #No
process_images(source_path, destination_path, sample_size)
