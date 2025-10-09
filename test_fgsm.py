import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from compute_gradcam import ResNet18  # Import ResNet18 from your existing implementation

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load precomputed FGSM dataset (10k clean + 10k perturbed)
fgsm_dataset = torch.load("cifar10_fgsm.pt")  # Ensure this file exists
test_loader = DataLoader(fgsm_dataset, batch_size=100, shuffle=False)

# Define model paths
model_paths = {
    'n0.5': 'fine_tuned_models/best_model_n_05.pt',
    'n0.7': 'fine_tuned_models/best_model_n_07.pt',
    'n0.9': 'fine_tuned_models/best_model_n_09.pt',
    'uniform': 'fine_tuned_models/best_model_uniform.pt'
}

# Evaluate model on FGSM dataset
def evaluate_model(model, loader):
    clean_correct = perturbed_correct = 0
    total_clean = total_perturbed = len(loader) // 2 * loader.batch_size

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            if batch_idx < len(loader) // 2:  # First half is clean data
                clean_correct += predicted.eq(targets).sum().item()
            else:  # Second half is perturbed data
                perturbed_correct += predicted.eq(targets).sum().item()

    clean_acc = 100 * clean_correct / total_clean
    perturbed_acc = 100 * perturbed_correct / total_perturbed
    return clean_acc, perturbed_acc

# Test all models on FGSM dataset
results = {}

for model_name, model_path in model_paths.items():
    # Load pretrained model
    model = ResNet18().to(device)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), model.fc)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Testing {model_name}...")

    # Evaluate on FGSM dataset
    clean_acc, fgsm_acc = evaluate_model(model, test_loader)

    results[model_name] = {
        'clean_acc': clean_acc,
        'fgsm_acc': fgsm_acc,
        'perf_drop': clean_acc - fgsm_acc
    }

# Print results
print("\n=== FGSM Model Comparison ===")
print("| Model     | Clean Acc (%) | FGSM Acc (%) | Performance Drop (%) |")
print("|-----------|---------------|--------------|-----------------------|")
for name, metrics in results.items():
    print(f"| {name:<9} | {metrics['clean_acc']:13.2f} | {metrics['fgsm_acc']:12.2f} | {metrics['perf_drop']:21.2f} |")
