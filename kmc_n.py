import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from tqdm import tqdm
import random
import time
import json
import os
import sys
import csv
from mpl_toolkits.mplot3d import Axes3D

# SIMULATION PARAMETERS
SIMULATION_PARAMETERS = {
    "simulation_name": "High_Temp_Smooth_Growth",
    "lattice_size": 50,
    "temperature": 700,
    "deposition_flux_per_site": 1000.0,
    "pre_exponential_factor": 1e12,
    "E_a_diffusion_base": 0.8,
    "E_a_lateral_binding": 0.3,
    "simulation_steps": 2000000,
    "data_snapshot_every_n_steps": 5000
}


class KMC_Simulation:
    def __init__(self, L, T, F, v0, E_a_diff_base, E_a_lateral):
        self.L = L
        self.T = T
        self.F = F
        self.v0 = v0
        self.E_a_diff_base = E_a_diff_base
        self.E_a_lateral = E_a_lateral
        self.k_B = 8.617333e-5
        self.lattice = np.zeros((L, L), dtype=int)
        self.time = 0.0
        self.total_deposition_events = 0
        self.neighbors_coords = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    def _get_rate(self, E_a):
        if self.T == 0: return 0.0
        return self.v0 * np.exp(-E_a / (self.k_B * self.T))

    def _get_neighbors_at(self, x, y):
        neighbors = []
        for dx, dy in self.neighbors_coords:
            nx = (x + dx) % self.L
            ny = (y + dy) % self.L
            neighbors.append((nx, ny))
        return neighbors

    def _get_diffusion_barrier(self, x, y):
        current_height = self.lattice[x, y]
        if current_height == 0:
            return None

        num_lateral_neighbors = 0
        is_embedded = True
        
        for nx, ny in self._get_neighbors_at(x, y):
            neighbor_height = self.lattice[nx, ny]
            
            if neighbor_height == current_height:
                num_lateral_neighbors += 1
            
            if neighbor_height < current_height:
                is_embedded = False

        if is_embedded:
            return None

        E_a = self.E_a_diff_base + (num_lateral_neighbors * self.E_a_lateral)
        return E_a

    def build_event_catalogue(self):
        events = []
        total_rate = 0.0

        deposition_rate = self.F * (self.L * self.L)
        events.append((deposition_rate, "DEPOSIT", None))
        total_rate += deposition_rate

        for x in range(self.L):
            for y in range(self.L):
                E_a = self._get_diffusion_barrier(x, y)
                if E_a is not None:
                    diff_rate = self._get_rate(E_a)
                    if diff_rate > 0:
                        events.append((diff_rate, "DIFFUSE", (x, y)))
                        total_rate += diff_rate

        return events, total_rate

    def select_event(self, events, total_rate):
        rho = random.uniform(0.0, total_rate)
        cumulative_rate = 0.0
        for (rate, event_type, coords) in events:
            cumulative_rate += rate
            if rho <= cumulative_rate:
                return event_type, coords
        return None, None

    def execute_event(self, event_type, coords):
        if event_type == "DEPOSIT":
            x, y = random.randint(0, self.L - 1), random.randint(0, self.L - 1)
            self.lattice[x, y] += 1
            self.total_deposition_events += 1
            
        elif event_type == "DIFFUSE":
            (x, y) = coords
            neighbor_coords = self._get_neighbors_at(x, y)
            neighbor_heights = [self.lattice[nx, ny] for nx, ny in neighbor_coords]
            min_height = min(neighbor_heights)
            
            if self.lattice[x, y] > min_height:
                target_sites = [
                    (nx, ny) for (nx, ny) in neighbor_coords 
                    if self.lattice[nx, ny] == min_height
                ]
                (tx, ty) = random.choice(target_sites)
                self.lattice[x, y] -= 1
                self.lattice[tx, ty] += 1

    def advance_time(self, total_rate):
        if total_rate == 0:
            return
        rho = random.uniform(0.0, 1.0)
        while rho == 0.0:
             rho = random.uniform(0.0, 1.0)
        dt = -np.log(rho) / total_rate
        self.time += dt

    def run_step(self):
        events, total_rate = self.build_event_catalogue()
        if total_rate == 0.0:
            return False
        event_type, coords = self.select_event(events, total_rate)
        if event_type is None:
             return True
        self.execute_event(event_type, coords)
        self.advance_time(total_rate)
        return True

    def get_rms_roughness(self):
        mean_height = np.mean(self.lattice)
        if mean_height == 0:
            return 0.0
        return np.sqrt(np.mean((self.lattice - mean_height)**2))

    def get_mean_height(self):
        return np.mean(self.lattice)

def run_simulation(params):
    print("Loading simulation parameters...")
    sim_name = params.get("simulation_name", "kmc_simulation")
    L = params["lattice_size"]
    T = params["temperature"]
    F = params["deposition_flux_per_site"]
    v0 = params["pre_exponential_factor"]
    E_a_diff = params["E_a_diffusion_base"]
    E_a_lat = params["E_a_lateral_binding"]
    steps = params["simulation_steps"]
    snapshot_freq = params["data_snapshot_every_n_steps"]

    print(f"Starting simulation: {sim_name}")

    output_dir = f"output_{sim_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "results.csv")
    lattice_file = os.path.join(output_dir, "final_lattice.npy")
    params_file = os.path.join(output_dir, "parameters_used.json")
    
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)

    sim = KMC_Simulation(L, T, F, v0, E_a_diff, E_a_lat)
    
    csv_headers = ["step", "time_s", "rms_roughness", "mean_height_ml"]
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
        print(f"Running simulation for {steps} steps...")
        start_time = time.time()

        for step in tqdm(range(steps)):
            if not sim.run_step():
                print("Simulation ended early.")
                break
                
            if step % snapshot_freq == 0:
                rms = sim.get_rms_roughness()
                height = sim.get_mean_height()
                writer.writerow([step, sim.time, rms, height])
        
        rms = sim.get_rms_roughness()
        height = sim.get_mean_height()
        writer.writerow([steps, sim.time, rms, height])

        end_time = time.time()
        print(f"\nSimulation finished in {end_time - start_time:.2f} seconds.")

    np.save(lattice_file, sim.lattice)
    print(f"Final lattice saved to: {lattice_file}")
    print(f"Results saved to: {results_file}")
    
    return output_dir

def plot_results(output_dir):
    print(f"\nGenerating plots for: {output_dir}")
    
    results_file = os.path.join(output_dir, "results.csv")
    lattice_file = os.path.join(output_dir, "final_lattice.npy")
    params_file = os.path.join(output_dir, "parameters_used.json")

    if not os.path.exists(results_file) or not os.path.exists(lattice_file):
        print(f"Error: Missing results.csv or final_lattice.npy in {output_dir}")
        return
        
    if not os.path.exists(params_file):
        sim_name = os.path.basename(output_dir)
    else:
        with open(params_file, 'r') as f:
            params = json.load(f)
            sim_name = params.get("simulation_name", os.path.basename(output_dir))

    data = pd.read_csv(results_file)
    lattice = np.load(lattice_file)
    data = data[data["time_s"] > 0].copy()
    if data.empty:
        print("No data with time > 0. Cannot generate log plots.")
        return

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    plt.plot(data["time_s"], data["rms_roughness"])
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Roughness (layers)")
    plt.title(f"RMS Roughness vs. Time\n({sim_name})")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "plot_roughness_vs_time.png")
    plt.savefig(plot_path)
    print(f"Saved: {plot_path}")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(data["time_s"], data["rms_roughness"], 'o', 
             label="KMC Data", markersize=3, alpha=0.6)
    
    try:
        log_time = np.log10(data["time_s"])
        log_rms = np.log10(data["rms_roughness"])
        
        fit_end_index = len(log_time) // 2
        if fit_end_index > 10:
            res = linregress(log_time.iloc[:fit_end_index], log_rms.iloc[:fit_end_index])
            beta = res.slope
            fit_line = 10**(res.intercept + res.slope * log_time)
            plt.plot(data["time_s"], fit_line, 'r--', 
                     label=f"Fit (First Half): $\\beta \\approx {beta:.3f}$")
            plt.text(0.05, 0.9, f"Growth Exponent $\\beta \\approx {beta:.3f}$", 
                     transform=plt.gca().transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.5))
    except Exception as e:
        print(f"Could not perform log-log fit: {e}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s) [Log Scale]")
    plt.ylabel("RMS Roughness (layers) [Log Scale]")
    plt.title(f"Roughness Growth (Log-Log Plot)\n({sim_name})")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "plot_roughness_growth_loglog.png")
    plt.savefig(plot_path)
    print(f"Saved: {plot_path}")
    plt.close()

    # --- 5. Plot 3: Final 3D Surface Topography ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    mean_h = np.mean(lattice)
    min_h, max_h = np.min(lattice), np.max(lattice)
    
    # Create X, Y coordinate grids
    X = np.arange(0, lattice.shape[1], 1)
    Y = np.arange(0, lattice.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    
    # Plot the surface
    ax.plot_surface(X, Y, lattice, cmap='viridis', rstride=1, cstride=1,
                      edgecolor='none', vmin=min_h, vmax=max_h)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Film Height (layers)')
    ax.set_zlim(0, max_h * 1.1) # Set z-limit to start from 0
                
    plt.title(f"Final 3D Surface Topography\n({sim_name})\n"
              f"Mean Height: {mean_h:.2f} layers | Roughness: {data['rms_roughness'].iloc[-1]:.2f} layers",
              fontsize=14)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "plot_final_topography.png")
    plt.savefig(plot_path)
    print(f"Saved: {plot_path}")
    plt.close()
    
    print("\nAll plots generated successfully.")

def main():
    try:
        import numpy, pandas, matplotlib, seaborn, scipy, tqdm
    except ImportError as e:
        print(f"Error: Missing required package: {e.name}")
        print("Please install the required packages:")
        print("pip install numpy pandas matplotlib seaborn scipy tqdm")
        sys.exit(1)
        
    output_directory = run_simulation(SIMULATION_PARAMETERS)
    if output_directory:
        plot_results(output_directory)

if __name__ == "__main__":
    main()
