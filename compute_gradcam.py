import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from noise import pnoise2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to generate Perlin noise
def generate_perlin_noise(shape, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0):
    """Generate a Perlin noise array with specified parameters."""
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
    noise = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize to [0, 1]
    return noise

# Define data transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

# Load CIFAR-10 train dataset
trainset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)

# Define ResNet-18 architecture (unchanged)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = nn.functional.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# Initialize and load the model
model = ResNet18().to(device)
model.load_state_dict(torch.load("PyTorch_CIFAR10-master/state_dicts/resnet18.pt", map_location=device))

# Set model to eval mode but ensure parameters require gradients for GradCAM
model.eval()
for param in model.parameters():
    param.requires_grad = True

# Define the target layer for Grad-CAM
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# Function to compute Grad-CAM for a batch of images
def compute_gradcam_batch(images, targets=None):
    # Don't use torch.no_grad() here since GradCAM needs gradients
    try:
        grayscale_cams = cam(input_tensor=images, targets=targets)
        return grayscale_cams
    except Exception as e:
        print(f"Error computing GradCAM: {e}")
        return None

# Function to apply adaptive Perlin noise
def apply_adaptive_perlin_noise(
    image, gradcam_map, noise_scale=0.5, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0
):
    """Apply Perlin noise to an image with intensity based on Grad-CAM values."""
    noise = generate_perlin_noise(image.shape[:2], scale, octaves, persistence, lacunarity)
    noise = np.stack([noise] * image.shape[2], axis=2)  # Expand to 3 channels
    adaptive_mask = 1.0 - 0.8 * np.expand_dims(gradcam_map, axis=2)  # Stronger noise reduction in key regions
    noisy_image = image + noise_scale * noise * adaptive_mask
    return np.clip(noisy_image, 0, 1)

# Process training set without plotting
def process_training_set(batch_size=64):
    os.makedirs("gradcam_values", exist_ok=True)
    gradcam_values = {}
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Computing Grad-CAM")):
        images = images.to(device)
        batch_gradcam = compute_gradcam_batch(images)
        
        if batch_gradcam is None:
            print(f"Skipping batch {batch_idx} due to GradCAM computation error")
            continue
            
        for i in range(images.size(0)):
            idx = batch_idx * batch_size + i
            if idx < len(trainset):
                gradcam_values[idx] = batch_gradcam[i]
                
        if (batch_idx + 1) % 100 == 0:
            with open(f"gradcam_values/gradcam_batch_{batch_idx}.pkl", "wb") as f:
                pickle.dump(gradcam_values, f)
            gradcam_values = {}
            
    if gradcam_values:
        with open(f"gradcam_values/gradcam_final.pkl", "wb") as f:
            pickle.dump(gradcam_values, f)
    print("Grad-CAM values computed and saved.")

# Demonstrate adaptive noise with three-image comparison
def demonstrate_adaptive_noise():
    # Get a sample image
    sample_image, sample_label = trainset[0]
    sample_image_input = sample_image.unsqueeze(0).to(device)
    
    # Compute Grad-CAM
    sample_gradcam = cam(input_tensor=sample_image_input)[0]
    
    sample_image_np = sample_image.numpy().transpose(1, 2, 0)
    sample_image_np = (sample_image_np * 0.5) + 0.5  # Denormalize to [0, 1]
    sample_image_np = np.clip(sample_image_np, 0, 1)

    # Regular Perlin noise
    regular_noise = generate_perlin_noise((32, 32), scale=10.0, octaves=6, persistence=0.5)
    regular_noise_3d = np.stack([regular_noise] * 3, axis=2)
    regular_noisy_image = np.clip(sample_image_np + 0.5 * regular_noise_3d, 0, 1)

    # Adaptive Perlin noise
    adaptive_noisy_image = apply_adaptive_perlin_noise(
        sample_image_np, sample_gradcam, noise_scale=0.5, scale=10.0, octaves=6, persistence=0.5
    )

    # Create a single plot with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    ax1.imshow(sample_image_np)
    ax1.set_title("Original Image")
    ax1.axis("off")
    ax2.imshow(regular_noisy_image)
    ax2.set_title("Regular Perlin Noise")
    ax2.axis("off")
    ax3.imshow(adaptive_noisy_image)
    ax3.set_title("Adaptive Perlin Noise")
    ax3.axis("off")
    plt.tight_layout()
    plt.savefig("perlin_noise_comparison.png")
    plt.show()

# Run the demonstration
if __name__ == "__main__":
    demonstrate_adaptive_noise()
    process_training_set()
else:
    # Only demonstrate if not imported as a module
    demonstrate_adaptive_noise()