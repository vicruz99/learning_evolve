# sol_000185 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5c6e3651) state=8bad8d42 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Set fixed seed for reproducibility
    rng = np.random.RandomState(42)
    n = 26

    # --- 1. Initialization ---
    # Initialize centers on a hexagonal lattice pattern with noise
    centers = np.zeros((n, 2))
    i = 0
    r_init = 0.02
    
    # Generate hexagonal points
    # Row spacing sqrt(3)/2 * diameter
    # We can generate more points than needed and pick the first 'n' that fit well
    points = []
    # Rough estimation: 6x5 grid of points
    for row in range(6):
        y = row * 0.18 + 0.05
        x_offset = (row % 2) * 0.1 + 0.05
        for col in range(6):
            x = x_offset + col * 0.20
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
    
    # If we have enough points, select 'n' of them
    if len(points) >= n:
        # Shuffle and pick n
        indices = rng.choice(len(points), n, replace=False)
        centers = np.array(points)[indices]
    else:
        # Fallback to random
        centers = rng.rand(n, 2) * 0.8 + 0.1
        
    # Add small random perturbation to break symmetry
    centers += rng.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, 0, 1)
    
    # Initial radii
    radii = np.full(n, r_init)

    # --- 2. Simulation Parameters ---
    max_iter = 6000
    dt = 0.008
    base_growth = 0.0004
    repulsion_k = 15.0
    boundary_k = 15.0

    # --- 3. Force-Directed Expansion Loop ---
    for step in range(max_iter):
        forces = np.zeros_like(centers)
        growth = base_growth * (1.0 - step / max_iter) # Cooling schedule
        
        # Boundary Forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Push away from walls if radius is too large or center is too close
            if x < r: forces[i, 0] += boundary_k * (r - x)
            if x > 1 - r: forces[i, 0] -= boundary_k * (x - (1 - r))
            if y < r: forces[i, 1] += boundary_k * (r - y)
            if y > 1 - r: forces[i, 1] -= boundary_k * (y - (1 - r))

        # Pairwise Repulsion Forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Prevent division by zero
                if dist < 1e-9:
                    dist = 1e-9
                    dx, dy = 1, 0 
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Force is proportional to overlap
                    f_mag = repulsion_k * overlap
                    
                    # Normalize direction
                    fx = (dx / dist) * f_mag
                    fy = (dy / dist) * f_mag
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Update Centers
        centers += forces * dt
        # Hard clip to ensure they don't escape the square during large forces
        centers = np.clip(centers, 0.0, 1.0)

        # Grow Radii
        radii += growth

    # --- 4. Post-Processing: Ensure Strict Validity ---
    # Radii might be slightly too large due to discrete steps.
    # We adjust them to the maximum valid radius given the final centers.
    
    for i in range(n):
        # Boundary constraints
        r_max = min(centers[i, 0], 1 - centers[i, 0], 
                    centers[i, 1], 1 - centers[i, 1])
        
        # Overlap constraints
        for j in range(n):
            if i != j:
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                # r_i <= dist - r_j
                r_ij = dist - radii[j]
                if r_ij < r_max:
                    r_max = r_ij
        
        # Apply the safe radius (with a tiny buffer for numerical safety)
        radii[i] = max(0, r_max - 1e-13)

    return centers, radii, float(np.sum(radii))
