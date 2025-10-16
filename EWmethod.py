#Edwards–Wilkinson (EW) model

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


L = 64
total_ML = 10
n_diff = 10          # Number of diffusion steps per atom


def compute_roughness(surface):
    h = surface - np.mean(surface)
    return np.sqrt(np.mean(h ** 2))

def deposit_atom(surface, L):
    x = np.random.randint(L)
    surface[x] += 1
    return x

def diffuse(surface, x, n_diff):
    #Atoms diffuse to lower neighboring sites
    for _ in range(n_diff):
        direction = np.random.choice([-1, 1])
        new_x = (x + direction) % len(surface)
        if surface[new_x] < surface[x]:
            surface[x] -= 1
            surface[new_x] += 1
            x = new_x
    return surface

def edwards_wilkinson(L, total_ML, n_diff):
    N_atoms = L * total_ML
    surface = np.zeros(L, dtype=int)
    roughness_evolution = []

    for i in trange(N_atoms, desc="Depositing atoms (EW)"):
        x = deposit_atom(surface, L)
        surface = diffuse(surface, x, n_diff)

        if i % (L // 2) == 0:
            roughness_evolution.append(compute_roughness(surface))

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(L), surface, color='orange')
    plt.title(f"Final Surface Morphology (EW, n_diff={n_diff})")
    plt.xlabel("Position (x)")
    plt.ylabel("Height")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(roughness_evolution, 'g-o')
    plt.title("Surface Roughness Evolution (EW)")
    plt.xlabel("Deposition progress")
    plt.ylabel("Roughness (RMS height)")
    plt.show()

    return surface, roughness_evolution

if __name__ == "__main__":
    surface, roughness = edwards_wilkinson(L, total_ML, n_diff)
    print(f"Final RMS roughness = {compute_roughness(surface):.3f}")

