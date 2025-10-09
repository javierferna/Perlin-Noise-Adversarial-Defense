import torch
import numpy as np
import os
import pickle
import random
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from noise import pnoise2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def generate_random_perlin_noise(shape):
    """Generate Perlin noise with random parameters within predefined ranges."""
    scale = random.uniform(5.0, 15.0)
    octaves = random.randint(4, 8)
    persistence = random.uniform(0.3, 0.7)
    lacunarity = random.uniform(1.5, 2.5)
    
    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            noise[i][j] = pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

def load_gradcam_values(directory='gradcam_values'):
    """Load all Grad-CAM values from pickle files."""
    all_gradcam_values = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as f:
                    gradcam_batch = pickle.load(f)
                    all_gradcam_values.update(gradcam_batch)
            except Exception as e:
                logging.error(f"Error loading {filepath}: {e}")
    logging.info(f"Loaded {len(all_gradcam_values)} Grad-CAM values")
    return all_gradcam_values

class AugmentedDataset(Dataset):
    """Dataset class with configurable noise adaptation."""
    def __init__(self, root='./data', gradcam_dir='gradcam_values', 
                 noise_scaling=0.7, use_uniform=False):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.original_dataset = datasets.CIFAR10(
            root=root, train=True, transform=self.transform, download=True
        )
        self.gradcam_maps = load_gradcam_values(gradcam_dir) if not use_uniform else {}
        self.noise_scaling = noise_scaling
        self.use_uniform = use_uniform
        self.total_size = len(self.original_dataset) * 2

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        if index < len(self.original_dataset):
            return self.original_dataset[index]
        
        original_idx = index - len(self.original_dataset)
        image, label = self.original_dataset[original_idx]
        
        # Generate Perlin noise
        noise = generate_random_perlin_noise((image.size(1), image.size(2)))
        noise_tensor = torch.from_numpy(noise).float()
        noise_tensor = (noise_tensor - 0.5) * 0.5  # Scale to [-0.25, 0.25]

        if not self.use_uniform and original_idx in self.gradcam_maps:
            gradcam_map = self.gradcam_maps[original_idx]
            gradcam_map = torch.from_numpy(gradcam_map).float()
            adaptive_mask = 1.0 - self.noise_scaling * gradcam_map
        else:
            adaptive_mask = torch.ones_like(noise_tensor)

        noise_tensor = noise_tensor * adaptive_mask
        perturbed_image = image + noise_tensor.unsqueeze(0).repeat(3, 1, 1)
        perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)
        
        return perturbed_image, label

def train_model(config_name, noise_scaling=0.7, use_uniform=False):
    """Train a model with specific noise configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\n{'='*40}\nTraining {config_name} configuration\n{'='*40}")
    
    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Create dataset (skip Grad-CAM for uniform)
    augmented_dataset = AugmentedDataset(
        noise_scaling=noise_scaling,
        use_uniform=use_uniform
    )
    
    # Split dataset
    train_size = int(0.8 * len(augmented_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        augmented_dataset, [train_size, len(augmented_dataset)-train_size]
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Initialize model
    from compute_gradcam import ResNet18
    model = ResNet18().to(device)
    model.load_state_dict(torch.load("PyTorch_CIFAR10-master/state_dicts/resnet18.pt", map_location=device))
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), model.fc)

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    best_model_path = f'fine_tuned_models/best_model_{config_name}.pt'
    
    start_time = time.time()
    
    for epoch in range(5):  # Train for 5 epochs as specified
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = sum(
            criterion(model(inputs.to(device)), targets.to(device)).item()
            for inputs, targets in val_loader
        ) / len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    end_time = time.time()
    
    training_time_hours = (end_time - start_time) / 3600
    
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    )
    
    logging.info(f"{config_name} Training Time: {training_time_hours:.2f} hours")
    logging.info(f"{config_name} Peak GPU Memory: {peak_gpu_memory_mb:.2f} MB")
    
def main():
    configurations = [
        ('n_05', 0.5, False),
        ('n_07', 0.7, False),
        ('n_09', 0.9, False),
        ('uniform', None, True),
    ]
    
    os.makedirs('fine_tuned_models', exist_ok=True)
    
    for config_name, scaling_factor, uniform_flag in configurations:
        train_model(config_name=config_name,
                    noise_scaling=scaling_factor,
                    use_uniform=uniform_flag)

if __name__ == "__main__":
   main()
