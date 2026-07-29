# sol_000032 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state eaaa636a) state=8338d744 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def compute_optimal_radii(centers, A_ub, pair_indices, boundary_indices, c, bounds):
    """Solves the LP to find optimal radii for fixed centers."""
    n = centers.shape[0]
    b_ub = np.zeros(A_ub.shape[0])
    idx = 0
    
    # Pairwise distance constraints
    for i, j in pair_indices:
        dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
        b_ub[idx] = dist
        idx += 1
        
    # Boundary clearance constraints
    for i in boundary_indices:
        x, y = centers[i]
        b_ub[idx] = min(x, 1.0 - x, y, 1.0 - y)
        idx += 1
        
    # linprog minimizes, so we passed c = -np.ones(n) to maximize sum of radii
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return res.x

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    
    # --- Precompute LP Constraint Structure ---
    A_ub_rows = []
    pair_indices = []
    boundary_indices = []
    
    # Pairwise constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i+1, n):
            pair_indices.append((i, j))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        boundary_indices.append(i)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        
    A_ub = np.array(A_ub_rows)
    c = -np.ones(n)  # Negative to maximize sum of radii
    bounds = [(0.0, None)] * n
    
    # --- Initialization ---
    centers = np.zeros((n, 2))
    idx = 0
    spacing = 0.13
    # Hexagonal grid layout
    for row in range(7):
        for col in range(7):
            if idx >= n: break
            x = spacing + col * spacing * 2.0
            y = spacing + row * spacing * 1.73205
            if col % 2 == 1:
                y += spacing * 0.866025
            centers[idx] = [x, y]
            idx += 1
        if idx >= n: break
        
    # Add noise to escape symmetric local minima
    centers += rng.normal(0, 0.02, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # --- Force-Directed Optimization ---
    lr = 0.018
    decay = 0.996
    steps = 1500
    
    for step in range(steps):
        radii = compute_optimal_radii(centers, A_ub, pair_indices, boundary_indices, c, bounds)
        
        forces = np.zeros_like(centers)
        
        # Coulomb repulsion to increase inter-circle distances
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                dist_sq = np.dot(diff, diff)
                dist = np.sqrt(dist_sq)
                if dist > 1e-5:
                    f = diff / (dist_sq * dist)  # Magnitude ~ 1/d^2
                    forces[i] += f
                    forces[j] -= f
                    
        # Center attraction to maximize distance from boundaries
        forces += 0.35 * (np.array([0.5, 0.5]) - centers)
        
        centers += lr * forces
        centers = np.clip(centers, 0.0, 1.0)
        
        lr *= decay
        
    # Final radius computation
    radii = compute_optimal_radii(centers, A_ub, pair_indices, boundary_indices, c, bounds)
    return centers, radii, np.sum(radii)
