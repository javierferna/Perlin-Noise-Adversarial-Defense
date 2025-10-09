import torch
import numpy as np
import random
import os
import logging
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from noise import pnoise2

# Configure logging
logging.basicConfig(
    filename='test_results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def generate_random_perlin_noise(shape):
    """Generate Perlin noise with completely random parameters"""
    scale = random.uniform(1.0, 20.0)
    octaves = random.randint(1, 10)
    persistence = random.uniform(0.1, 1.0)
    lacunarity = random.uniform(1.0, 3.0)
    
    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            noise[i][j] = pnoise2(
                i/scale, j/scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity
            )
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

class UniformNoiseTestDataset(Dataset):
    """Dataset with 10k clean and 10k uniformly perturbed test images"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.total_size = len(base_dataset) * 2  # 20k total

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        if index < len(self.base_dataset):
            return self.base_dataset[index]
        
        original_idx = index - len(self.base_dataset)
        image, label = self.base_dataset[original_idx]
        
        # Generate completely random Perlin noise
        noise = generate_random_perlin_noise((32, 32))
        noise_tensor = torch.from_numpy(noise).float()
        noise_tensor = (noise_tensor - 0.5) * 0.5  # Scale to [-0.25, 0.25]
        
        # Apply uniform noise to all regions
        perturbed_image = image + noise_tensor.unsqueeze(0).repeat(3, 1, 1)
        perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)
        
        return perturbed_image, label

def evaluate_model(model, test_loader, device):
    """Evaluate model on clean and perturbed test sets"""
    clean_correct = perturbed_correct = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Split clean and perturbed
            if batch_idx < 100:  # First 100 batches are clean (10k images)
                clean_correct += predicted.eq(targets).sum().item()
            else:  # Last 100 batches are perturbed (10k images)
                perturbed_correct += predicted.eq(targets).sum().item()

    return clean_correct, perturbed_correct

def test_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Define models to test
    model_paths = {
        'n0.5': 'fine_tuned_models/best_model_n_05.pt',
        'n0.7': 'fine_tuned_models/best_model_n_07.pt',
        'n0.9': 'fine_tuned_models/best_model_n_09.pt',
        'uniform': 'fine_tuned_models/best_model_uniform.pt'
    }

    # Create test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    base_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_dataset = UniformNoiseTestDataset(base_test)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    results = {}
    
    for model_name, model_path in model_paths.items():
        try:
            # Load model
            from compute_gradcam import ResNet18  # Update with your actual import
            model = ResNet18()
            model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), model.fc)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device).eval()
            
            logging.info(f"Testing {model_name}...")
            
            # Evaluate
            clean_correct, perturbed_correct = evaluate_model(model, test_loader, device)
            
            # Calculate metrics
            clean_acc = 100 * clean_correct / 10000
            perturbed_acc = 100 * perturbed_correct / 10000
            perf_drop = clean_acc - perturbed_acc
            
            results[model_name] = {
                'clean_acc': clean_acc,
                'perturbed_acc': perturbed_acc,
                'perf_drop': perf_drop
            }
            
            logging.info(f"{model_name} results:")
            logging.info(f"  Clean Accuracy: {clean_acc:.2f}%")
            logging.info(f"  Perturbed Accuracy: {perturbed_acc:.2f}%")
            logging.info(f"  Performance Drop: {perf_drop:.2f}%")
            
        except Exception as e:
            logging.error(f"Error testing {model_name}: {e}")

    # Print comparison table
    logging.info("\nModel Comparison:")
    logging.info("| Model    | Clean Acc (%) | Perturbed Acc (%) | Performance Drop (%) |")
    logging.info("|----------|---------------|--------------------|-----------------------|")
    for name, metrics in results.items():
        logging.info(f"| {name:<7} | {metrics['clean_acc']:13.2f} | {metrics['perturbed_acc']:18.2f} | {metrics['perf_drop']:19.2f} |")

if __name__ == "__main__":
    logging.info("Starting comprehensive model testing...")
    try:
        test_all_models()
        logging.info("Testing completed successfully.")
    except Exception as e:
        logging.error(f"Critical error during testing: {e}")
