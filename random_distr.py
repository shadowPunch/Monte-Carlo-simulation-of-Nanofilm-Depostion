
#Monte Carlo simulation of thin film growth using the Random Deposition (RD) model.


import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


L = 64              # Substrate length
total_ML = 10       # Total monolayers



def compute_roughness(surface):
    #RMS surface roughness
    h = surface - np.mean(surface)
    return np.sqrt(np.mean(h ** 2))

def random_deposition(L, total_ML):
    #Random Deposition-no diffusion
    N_atoms = L * total_ML
    surface = np.zeros(L, dtype=int)
    roughness_evolution = []

    for i in trange(N_atoms, desc="Depositing atoms (RD)"):
        x = np.random.randint(L)
        surface[x] += 1

        if i % (L // 2) == 0:
            roughness_evolution.append(compute_roughness(surface))

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(L), surface, color='steelblue')
    plt.title("Final Surface Morphology (Random Deposition)")
    plt.xlabel("Position (x)")
    plt.ylabel("Height")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(roughness_evolution, 'r-o')
    plt.title("Surface Roughness Evolution (RD)")
    plt.xlabel("Deposition progress")
    plt.ylabel("Roughness (RMS height)")
    plt.show()

    return surface, roughness_evolution

# --- Run Simulation ---
if __name__ == "__main__":
    surface, roughness = random_deposition(L, total_ML)
    print(f"Final RMS roughness = {compute_roughness(surface):.3f}")

