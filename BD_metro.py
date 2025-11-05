import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


Lx, Ly = 60, 60          # Surface dimensions
steps = 30000            

# --- Metropolis Parameters ---
kT = 1.0                 
J = 1.0                  
diffusion_per_deposition = 10 

h = np.zeros((Lx, Ly))   # Height field
rng = np.random.default_rng()

# Data tracking arrays
roughness_data = []
avg_height_data = []
time_steps = []

def local_energy(x, y):
    """Calculates the EW (curvature) energy for a single site."""
    left = h[(x - 1) % Lx, y]
    right = h[(x + 1) % Lx, y]
    up = h[x, (y - 1) % Ly]
    down = h[x, (y + 1) % Ly]
    
    
    return J * ((h[x,y]-left)**2 + (h[x,y]-right)**2 + (h[x,y]-up)**2 + (h[x,y]-down)**2)


fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly))
colormap = cm.viridis

def plot_surface(step):
    """Updates the 3D surface plot."""
    ax.clear()
    surf = ax.plot_surface(X, Y, h.T, cmap=colormap, linewidth=0, antialiased=False)
    
    max_h = np.max(h)
    min_h = np.min(h)
    if max_h == min_h: max_h += 1
        
    ax.set_zlim(min_h - 1, max_h + 1)
    ax.set_title(f"2D Ballistic + Metropolis (kT={kT})\nStep = {step}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    plt.pause(0.01)

print(f"Running 2D Ballistic + Metropolis (kT={kT}, J={J}) Simulation...")
for step in range(steps):
    
    

    x_dep, y_dep = rng.integers(0, Lx), rng.integers(0, Ly)
    

    current_h = h[x_dep, y_dep]
    left      = h[(x_dep - 1) % Lx, y_dep]
    right     = h[(x_dep + 1) % Lx, y_dep]
    up        = h[x_dep, (y_dep - 1) % Ly]
    down      = h[x_dep, (y_dep + 1) % Ly]
    
    max_neighbor_h = max(current_h, left, right, up, down)
   
    h[x_dep, y_dep] = max_neighbor_h + 1
            
    for _ in range(diffusion_per_deposition):

        x_diff, y_diff = rng.integers(0, Lx), rng.integers(0, Ly)
        if h[x_diff, y_diff] == 0:
            continue # Can't move an atom from an empty site

        # Pick a random neighbor to hop to
        neighbors = [
            ((x_diff - 1) % Lx, y_diff), ((x_diff + 1) % Lx, y_diff),
            (x_diff, (y_diff - 1) % Ly), (x_diff, (y_diff + 1) % Ly)
        ]
        nx, ny = neighbors[rng.integers(0, 4)]

       
        E_before = local_energy(x_diff, y_diff) + local_energy(nx, ny)
        
        # 2. Perform the hop temporarily
        h[x_diff, y_diff] -= 1
        h[nx, ny] += 1
        
        # 3. Get energy after the hop
        E_after = local_energy(x_diff, y_diff) + local_energy(nx, ny)
        
      
        dE = E_after - E_before
        
      
        if dE > 0 and rng.random() > np.exp(-dE / kT):
          
            h[x_diff, y_diff] += 1
            h[nx, ny] -= 1

    if step % (steps // 100) == 0 or step == steps - 1:
        mean_h = np.mean(h)

        w = np.sqrt(np.mean((h - mean_h)**2))
        avg_height_data.append(mean_h)
        roughness_data.append(w)
        time_steps.append(step)
        
        if step % (steps // 10) == 0:
             print(f"Step {step}/{steps} | Roughness: {w:.3f}")

    # --- Visualization ---
    if step % (steps // 10) == 0:
        plot_surface(step)


print("Simulation complete. Showing final surface.")
plot_surface(steps)
plt.show()


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(time_steps, roughness_data, color='red')
plt.title("Surface Roughness (RMS) vs Time")
plt.xlabel("Monte Carlo Steps (Atoms Deposited)")
plt.ylabel("Roughness (w)")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(time_steps, avg_height_data, color='blue')
plt.title("Average Height vs Time")
plt.xlabel("Monte Carlo Steps (Atoms Deposited)")
plt.ylabel("Mean Height")
plt.grid(True)

plt.tight_layout()
plt.show()


print(f"Final average height: {avg_height_data[-1]:.3f}")
print(f"Final surface roughness (RMS): {roughness_data[-1]:.3f}")

