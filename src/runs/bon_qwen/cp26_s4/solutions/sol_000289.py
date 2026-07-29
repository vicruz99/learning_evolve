# sol_000289 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e8a963c) state=7cabb85b sum of radii=1.391223 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def compute_optimal_radii(centers):
    """
    Solve LP to find maximum radii for fixed centers.
    Constraints: r_i + r_j <= dist(i,j), r_i <= dist_to_wall(i), r_i >= 0
    """
    n = centers.shape[0]
    c = -np.ones(n)  # Maximize sum radii -> Minimize negative sum
    A_ub = []
    b_ub = []
    
    # Wall constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    for i in range(n):
        x, y = centers[i]
        for val in (x, 1-x, y, 1-y):
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(val)
            
    # Pairwise constraints: r_i + r_j <= distance(i, j)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    # Solve LP
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                  bounds=[(0, None)]*n, method='highs', options={'disp': False})
    
    if res.success:
        return res.x
    # Fallback if LP fails (should not happen with valid centers)
    return np.full(n, 1e-7)

def run_packing():
    np.random.seed(42)
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Run multiple trials from different initial configurations
    for trial in range(4):
        centers = np.zeros((n, 2))
        idx = 0
        # Initialize with a hexagonal lattice pattern
        spacing = 0.22
        for r in range(7):
            for c in range(7):
                if idx >= n: break
                x = c * spacing + (r % 2) * spacing / 2
                y = r * spacing * np.sqrt(3) / 2
                centers[idx] = [x + 0.1, y + 0.1]
                idx += 1
                
        # Add random perturbation to break symmetry and escape local traps
        centers += np.random.uniform(-0.15, 0.15, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        lr = 0.01
        for step in range(1200):
            # Step 1: Find optimal radii for current positions
            radii = compute_optimal_radii(centers)
            curr_sum = np.sum(radii)
            
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
                
            # Step 2: Compute repulsive forces to improve positions
            forces = np.zeros_like(centers)
            for i in range(n):
                for j in range(i+1, n):
                    diff = centers[i] - centers[j]
                    dist = np.sqrt(np.sum(diff**2)) + 1e-9
                    tightness = (radii[i] + radii[j]) / dist
                    
                    # Force pushes centers apart, stronger when constraints are tight
                    # Added baseline repulsion (0.1) to prevent stagnation
                    f_mag = (tightness + 0.1) / dist
                    forces[i] += f_mag * diff
                    forces[j] -= f_mag * diff
                    
            # Update positions
            centers += lr * forces
            centers = np.clip(centers, 0.0, 1.0)
            lr *= 0.997  # Decay learning rate for stability
            
            # Occasional noise to escape local minima
            if step % 250 == 0 and step > 0:
                centers += np.random.uniform(-0.02, 0.02, centers.shape)
                centers = np.clip(centers, 0.05, 0.95)
                
    if best_centers is None:
        best_centers = np.random.rand(n, 2) * 0.8 + 0.1
        best_radii = compute_optimal_radii(best_centers)
        
    # Apply tiny shrink for numerical safety against 1e-12 validation tolerance
    best_radii *= 0.9999999
    return best_centers, best_radii, np.sum(best_radii)
