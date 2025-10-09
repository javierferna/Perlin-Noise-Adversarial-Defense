import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from resnet import resnet18  # Import ResNet-18 from your resnet.py

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Correct CIFAR-10 normalization values
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Load CIFAR-10 FGSM dataset (10k clean + 10k perturbed)
fgsm_dataset = torch.load("cifar10_fgsm.pt")  # Ensure this file exists
test_loader = DataLoader(fgsm_dataset, batch_size=100, shuffle=False)

# Load pretrained ResNet-18 model
model = resnet18(pretrained=False).to(device)
model.load_state_dict(torch.load("resnet18.pt", map_location=device))
model.eval()

# Evaluate model on FGSM dataset
def evaluate_model(model, loader):
    clean_correct = perturbed_correct = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            if batch_idx < len(loader) // 2:  # First half is clean data
                clean_correct += predicted.eq(targets).sum().item()
            else:  # Second half is perturbed data
                perturbed_correct += predicted.eq(targets).sum().item()
    
    clean_acc = 100 * clean_correct / (len(loader) // 2 * loader.batch_size)
    perturbed_acc = 100 * perturbed_correct / (len(loader) // 2 * loader.batch_size)
    return clean_acc, perturbed_acc

# Run evaluation
clean_acc, fgsm_acc = evaluate_model(model, test_loader)

print(f"Clean Accuracy: {clean_acc:.2f}%")
print(f"FGSM Attack Accuracy: {fgsm_acc:.2f}%")
print(f"Performance Drop: {clean_acc - fgsm_acc:.2f}%")
