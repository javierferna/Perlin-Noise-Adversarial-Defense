import numpy as np
import matplotlib.pyplot as plt
from compute_gradcam import generate_perlin_noise  

def main():
    # Image size (CIFAR‑10 style 32×32 is plenty for illustration)
    shape = (32, 32)

    # (title, kwargs) tuples for the four panels
    configs = [
        ("Octaves=1, Scale=10.0",               dict(scale=10.0, octaves=1,  persistence=0.5, lacunarity=2.0)),
        ("Octaves=6, Scale=10.0",               dict(scale=10.0, octaves=6,  persistence=0.5, lacunarity=2.0)),
        ("Octaves=6, Scale=5.0",                dict(scale=5.0,  octaves=6,  persistence=0.5, lacunarity=2.0)),
        ("Octaves=6, Scale=10.0, Persistence=0.8",
                                               dict(scale=10.0, octaves=6,  persistence=0.8, lacunarity=2.0)),
    ]

    # Build the figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for ax, (title, params) in zip(axes, configs):
        noise = generate_perlin_noise(shape, **params)
        ax.imshow(noise, cmap="viridis", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("perlin_parameter_sweep.png", dpi=300)
    plt.show()
    print("Figure saved to perlin_parameter_sweep.png")

if __name__ == "__main__":
    main()