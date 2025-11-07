import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


Lx, Ly = 60, 60          # Surface dimensions
steps = 30000            # Number of atoms to deposit
n_diff = 10              # Number of diffusion hops to attempt per atom
h = np.zeros((Lx, Ly))   # Height field

rng = np.random.default_rng()


roughness_data = []
avg_height_data = []
time_steps = []


fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly))
colormap = cm.viridis

def plot_surface(step):
    """Updates the 3D surface plot."""
    ax.clear()
    surf = ax.plot_surface(X, Y, h.T, cmap=colormap, linewidth=0, antialiased=False)
    
    # Set a reasonable Z limit, handling the case of a flat surface
    max_h = np.max(h)
    min_h = np.min(h)
    if max_h == min_h:
        max_h += 1
        
    ax.set_zlim(min_h - 1, max_h + 1)
    ax.set_title(f"2D EW Growth (Simplified Model)\nStep = {step}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    plt.pause(0.01) 

print("Running 2D Simplified EW Simulation...")
for step in range(steps):
    
 
    x, y = rng.integers(0, Lx), rng.integers(0, Ly)
    h[x, y] += 1
    
    current_x, current_y = x, y
    
    for _ in range(n_diff):
        # Get all 4 neighbors with periodic boundary conditions
        neighbors = [
            ((current_x - 1) % Lx, current_y),  
            ((current_x + 1) % Lx, current_y),  
            (current_x, (current_y - 1) % Ly),  
            (current_x, (current_y + 1) % Ly)   
        ]
        
        nx, ny = neighbors[rng.integers(0, 4)]
        
        if h[nx, ny] < h[current_x, current_y]:
            # Move the atom
            h[current_x, current_y] -= 1
            h[nx, ny] += 1
            
            # Update the atom's current position for the next hop attempt
            current_x, current_y = nx, ny
            
 
    if step % (steps // 100) == 0 or step == steps - 1:
        mean_h = np.mean(h)
        w = np.sqrt(np.mean((h - mean_h)**2))
        avg_height_data.append(mean_h)
        roughness_data.append(w)
        time_steps.append(step)
        
        if step % (steps // 10) == 0:
             print(f"Step {step}/{steps} | Roughness: {w:.3f}")

 
    if step % (steps // 10) == 0:
        plot_surface(step)


print("Simulation complete. Showing final surface.")
plot_surface(steps)
plt.show()


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(time_steps, roughness_data, color='red')
plt.title("Surface Roughness vs Time")
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
print(f"Final surface roughness: {roughness_data[-1]:.3f}")

