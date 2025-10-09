import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-10 normalization
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Load test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_dataset = datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Load pretrained model
from resnet import resnet18
model = resnet18().to(device)
model.load_state_dict(torch.load("resnet18.pt", map_location=device))
model.eval()

# FGSM attack parameters
epsilon = 0.03

def fgsm_attack(images, labels):
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Generate perturbation
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    
    # Clip to valid range
    perturbed_images = torch.clamp(perturbed_images, -2.0, 2.0)
    
    return perturbed_images.detach()

# Track resources
start_time = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# Generate adversarial examples
original_images = []
adversarial_images = []
labels = []

for batch in DataLoader(test_dataset, batch_size=100, shuffle=False):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Generate adversarial batch
    perturbed = fgsm_attack(inputs, targets)
    
    # Store results
    original_images.append(inputs.cpu())
    adversarial_images.append(perturbed.cpu())
    labels.append(targets.cpu())

# Combine datasets
original_images = torch.cat(original_images)
adversarial_images = torch.cat(adversarial_images)
labels = torch.cat(labels)

# Create final dataset (10k original + 10k adversarial)
combined_images = torch.cat([original_images, adversarial_images])
combined_labels = torch.cat([labels, labels])

# Create dataset object
final_dataset = TensorDataset(combined_images, combined_labels)

# Calculate metrics
execution_time = time.time() - start_time
memory_usage = torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0

print("\n=== Dataset Creation Metrics ===")
print(f"Total observations: {len(final_dataset):,}")
print(f"Original/Adversarial ratio: {len(test_dataset)}/{len(test_dataset)}")
print(f"Execution time: {execution_time:.2f} seconds")
print(f"Peak GPU memory: {memory_usage:.2f} MB")

# Save dataset
torch.save(final_dataset, "cifar10_fgsm.pt")
print("\nSaved dataset to cifar10_fgsm.pt")