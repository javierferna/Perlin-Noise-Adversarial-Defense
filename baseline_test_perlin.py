import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from noise import pnoise2                           
from resnet import resnet18                        

def generate_random_perlin_noise(shape):
    """Random 32×32 Perlin tile in [0,1]."""
    scale       = np.random.uniform(1.0, 20.0)
    octaves     = np.random.randint(1, 10)
    persistence = np.random.uniform(0.1, 1.0)
    lacunarity  = np.random.uniform(1.0, 3.0)

    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            noise[i, j] = pnoise2(i / scale, j / scale,
                                   octaves=octaves,
                                   persistence=persistence,
                                   lacunarity=lacunarity)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

# ----------------------------------------------------------------------
class PerlinNoiseCIFAR10(Dataset):
    """Returns 10 k clean + 10 k Perlin‑perturbed test images."""
    def __init__(self, base_ds):
        self.base = base_ds
        self.total = len(base_ds) * 2

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < len(self.base):
            return self.base[idx]               # clean
        # perturbed
        img, label = self.base[idx - len(self.base)]
        n = generate_random_perlin_noise((32, 32))
        n = (torch.from_numpy(n).float() - 0.5) * 0.5           # → [‑0.25,0.25]
        img_adv = torch.clamp(img + n.unsqueeze(0).repeat(3,1,1), -1.0, 1.0)
        return img_adv, label

# ----------------------------------------------------------------------
def evaluate(model, loader, device):
    clean = adv = 0
    half  = len(loader) // 2
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            if i < half:  clean += pred.eq(y).sum().item()
            else:         adv   += pred.eq(y).sum().item()
    return 100*clean/10000, 100*adv/10000   # CIFAR‑10 test set size

# ----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load model exactly as saved (no extra Dropout)
    model = resnet18().to(device)
    model.load_state_dict(torch.load("resnet18.pt", map_location=device))
    model.eval()

    # CIFAR‑10 test set
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,)*3, (0.5,)*3)])
    base_test = datasets.CIFAR10(root="./data", train=False,
                                 download=True, transform=tfm)

    # Build clean+Perlin dataset & loader
    test_ds = PerlinNoiseCIFAR10(base_test)
    loader  = DataLoader(test_ds, batch_size=100, shuffle=False, num_workers=4)

    acc_clean, acc_perlin = evaluate(model, loader, device)
    print(f"\nBaseline ResNet‑18:")
    print(f"  • Clean accuracy .............. {acc_clean:5.2f} %")
    print(f"  • Perlin‑noise accuracy ....... {acc_perlin:5.2f} %")
    print(f"  • Performance drop ............ {acc_clean - acc_perlin:5.2f} %")

if __name__ == "__main__":
    main()