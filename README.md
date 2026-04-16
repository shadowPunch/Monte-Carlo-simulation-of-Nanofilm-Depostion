# Monte Carlo Simulation of Thin Film Deposition and Growth

A hierarchical computational physics framework simulating nanofilm growth using Monte Carlo methods — from simple stochastic models to fully material-specific atomistic simulations.

**Author:** Nithish Ravikkumar (Enrollment No. 23123028)
**Code:** https://github.com/shadowPunch/Monte-Carlo-simulation-of-Nanofilm-Depostion

---

## Overview

This project models thin film deposition at the nanoscale using a progression of Monte Carlo techniques. Starting from the simplest random deposition baseline, the framework builds up to a material-specific Kinetic Monte Carlo (KMC) simulation incorporating temperature-dependent diffusion, Arrhenius kinetics, and realistic activation energy barriers.

The goal is to understand how atomic-scale mechanisms — random flux, surface diffusion, lateral growth, and geometric shadowing — collectively determine surface morphology and roughness in thin film systems relevant to semiconductors, optical coatings, and magnetic storage media.

---

## Models Implemented

### 1. Random Deposition (RD)
The baseline model. Particles deposit at random columns with no diffusion or neighbor interaction. Each column grows independently, following Poisson statistics. Predicted growth exponent: beta = 0.5.

### 2. Edwards-Wilkinson (EW)
Introduces surface relaxation via downhill diffusion after each deposition event. Atoms hop to lower-energy neighboring sites, smoothing the surface. Implements the linear EW equation in both 1D and 2D. Predicted growth exponent: beta = 0.25.

### 3. Kardar-Parisi-Zhang (KPZ)
Extends EW with a non-linear lateral growth term. Particles preferentially deposit at the edges of mounds, growing perpendicular to the local surface slope. Implemented in 1D and 2D. Predicted growth exponent: beta = 0.33.

### 4. Ballistic Deposition (BD) with Metropolis Relaxation
A hybrid 2D model combining geometric shadowing (ballistic aggregation) with thermally-activated surface relaxation via the Metropolis algorithm. Captures the roughening-vs-smoothing competition characteristic of physical vapor deposition.

### 5. Material-Specific Kinetic Monte Carlo (KMC)
The most advanced model. Uses the Bortz-Kalos-Lebowitz (BKL) algorithm to simulate rare barrier-crossing events via Arrhenius-rate-weighted event selection. Incorporates temperature-dependent diffusion, material-specific activation energies, lateral binding energies, and realistic deposition flux. Tracks RMS roughness and mean film height over time.

---

## Growth Exponents Summary

| Model | Dimensions | Predicted beta | Simulated beta |
|-------|------------|----------------|----------------|
| RD    | 1D         | 0.50           | ~0.50          |
| EW    | 1D         | 0.25           | ~0.25          |
| KPZ   | 1D         | 0.33           | ~0.33          |
| EW    | 2D         | ~0 (log)       | near-log       |
| BD    | 2D         | T-dependent    | regime-dependent |
| KMC   | 2D         | material-dependent | material-dependent |

---

## Physics Background

**RMS Surface Roughness**

    w(L,t) = sqrt( (1/L^d) * sum_x [ h(x,t) - h_mean(t) ]^2 )

**Arrhenius Diffusion Rate (KMC)**

    r = v0 * exp( -Ea / kB*T )

where v0 ~ 10^12 s^-1 is the attempt frequency and Ea is the activation energy barrier.

**Metropolis Acceptance Criterion (BD)**

    Accept if dE <= 0
    Accept with probability exp(-dE / kT) if dE > 0

**KPZ Equation**

    dh/dt = v * nabla^2(h) + (lambda/2) * (nabla h)^2 + eta(x,t)

**EW Equation**

    dh/dt = v * nabla^2(h) + eta(x,t)

---

## KMC Temperature Regimes

| Temperature   | Diffusion Regime     | Film Morphology              |
|---------------|----------------------|------------------------------|
| T < 400 K     | Kinetically limited  | Rough, columnar (BD-like)    |
| 400-800 K     | Intermediate         | Island nucleation, moderate  |
| T > 800 K     | Thermodynamically limited | Smooth, compact (EW-like) |

---

## Key Parameters

| Parameter         | Description                                    |
|-------------------|------------------------------------------------|
| L                 | Substrate lattice size                         |
| N_atoms           | Total atoms deposited                          |
| n_diff            | Diffusion attempts per deposition step (EW/KPZ)|
| p_side            | Lateral growth probability (KPZ)               |
| kT                | Thermal energy for Metropolis acceptance (BD)  |
| J                 | Surface tension strength (BD)                  |
| T (Kelvin)        | Substrate temperature (KMC)                    |
| F                 | Deposition flux — atoms per site per second (KMC) |
| v0                | Pre-exponential attempt frequency (KMC)        |
| Ea_base           | Base diffusion activation barrier in eV (KMC) |
| Ea_lateral        | Additional barrier per lateral neighbor (KMC)  |

---

## Dependencies

    pip install numpy matplotlib scipy

| Package      | Purpose                              |
|--------------|--------------------------------------|
| numpy        | Array operations and random sampling |
| matplotlib   | Surface topology and roughness plots |
| scipy        | Optional analysis utilities          |

---

## Project Structure

    ├── random_deposition.py       # 1D RD model
    ├── edwards_wilkinson.py       # 1D and 2D EW model
    ├── kardar_parisi_zhang.py     # 1D and 2D KPZ model
    ├── ballistic_deposition.py    # 2D BD + Metropolis relaxation
    ├── kinetic_monte_carlo.py     # Material-specific KMC (BKL algorithm)
    ├── analysis.py                # Roughness tracking and exponent fitting
    └── Report.pdf                 # Full project report with results

---

## Results Summary

All 1D growth exponents match theoretical predictions from scaling theory. Key findings:

- RD produces the roughest surfaces (beta ~ 0.5); EW the smoothest (beta ~ 0.25) due to diffusive relaxation.
- In 2D, the EW model exhibits near-logarithmic roughness growth, reflecting more effective relaxation with four neighbors vs two in 1D.
- The BD + Metropolis hybrid reproduces realistic PVD microstructures: high-roughness columnar at low kT, smooth dense films at high kT.
- KMC successfully distinguishes material classes: low-Ea materials (noble metals) grow smooth compact films; high-Ea materials produce rough porous morphologies.

---

## Limitations and Future Work

- The KMC lattice is rigid and crystalline — amorphous growth is not supported.
- Desorption events are not yet implemented, limiting applicability to CVD scenarios.
- Direct comparison with experimental AFM roughness data would provide quantitative validation.
- Extension to oblique-angle deposition for studying anisotropic columnar growth.

---

## References

1. Sasamoto & Spohn, "One-dimensional KPZ equation: An exact solution and its universality," Phys. Rev. Lett., 104, 230602, 2010.
2. Metropolis et al., "Equation of State Calculations by Fast Computing Machines."
3. Sasamoto, "The 1D KPZ equation: Height distribution and universality," PTEP.
4. Bukkuru et al., "Simulation of 2D ballistic deposition of porous nanostructured thin-films."
