#Kardar–Parisi–Zhang (KPZ) model

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

L = 64
total_ML = 10
p_side = 0.4         # Probability of lateral sticking (controls non-linearity)


def compute_roughness(surface):
    h = surface - np.mean(surface)
    return np.sqrt(np.mean(h ** 2))

def kpz_growth(L, total_ML, p_side=0.4):
    #KPZ-like stochastic growth with lateral sticking
    N_atoms = L * total_ML
    surface = np.zeros(L, dtype=int)
    roughness_evolution = []

    for i in trange(N_atoms, desc="Depositing atoms (KPZ)"):
        x = np.random.randint(L)

        # Lateral growth probability (slope-dependent)
        left, right = (x - 1) % L, (x + 1) % L

        # If side site is higher, stick to its side with probability p_side
        if np.random.rand() < p_side:
            if surface[left] > surface[x]:
                surface[x] = surface[left]
            elif surface[right] > surface[x]:
                surface[x] = surface[right]

        # Deposit atom
        surface[x] += 1

        if i % (L // 2) == 0:
            roughness_evolution.append(compute_roughness(surface))

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(L), surface, color='crimson')
    plt.title(f"Final Surface Morphology (KPZ, p_side={p_side})")
    plt.xlabel("Position (x)")
    plt.ylabel("Height")
    plt.show()
    
    plt.figure(figsize=(6, 4))
    plt.plot(roughness_evolution, 'b-o')
    plt.title("Surface Roughness Evolution (KPZ)")
    plt.xlabel("Deposition progress")
    plt.ylabel("Roughness (RMS height)")
    plt.show()

    return surface, roughness_evolution

if __name__ == "__main__":
    surface, roughness = kpz_growth(L, total_ML, p_side)
    print(f"Final RMS roughness = {compute_roughness(surface):.3f}")

