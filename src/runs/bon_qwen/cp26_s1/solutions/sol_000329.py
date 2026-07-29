# sol_000329 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 28c61761) state=9b05a36d sum of radii=1.530600 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    N = 26

    # --- 1. Initialization ---
    # Start with a perturbed hexagonal lattice for high initial density
    centers = np.zeros((N, 2))
    row = 0
    col = 0
    # Hexagonal spacing parameters
    dx = 0.22
    dy = dx * np.sqrt(3) / 2
    
    idx = 0
    while idx < N:
        x = 0.15 + col * dx + (0.5 * dx if row % 2 == 1 else 0)
        y = 0.15 + row * dy
        # Wrap around if out of bounds to maintain density
        if x > 0.85: 
            col += 1
            row += 1
            col = 0
            continue
        if y > 0.85:
            break
            
        # Add random perturbation to break symmetry
        centers[idx] = [x + np.random.uniform(-0.01, 0.01), 
                        y + np.random.uniform(-0.01, 0.01)]
        idx += 1
        col += 1
    
    # --- 2. Optimization Loop ---
    for _ in range(50): # Iterate 50 times
        
        # Step A: Solve for optimal radii given fixed centers (LP)
        # Objective: Minimize -sum(r)
        c_obj = np.ones(N) * -1
        
        # Constraints: r_i + r_j <= dist_ij
        # Matrix A_ub @ r <= b_ub
        A_ub = []
        b_ub = []
        
        # Pairwise constraints
        dists = cdist(centers, centers)
        for i in range(N):
            for j in range(i + 1, N):
                row = np.zeros(N)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dists[i, j])
        
        # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        # Handled via bounds in linprog
        bounds_r = []
        for i in range(N):
            x, y = centers[i]
            max_r = min(x, 1 - x, y, 1 - y)
            bounds_r.append((0, max_r))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
            if not res.success:
                break
            radii = res.x
        except ValueError:
            break

        # Step B: Adjust centers to relieve pressure
        # Simple repulsion force: if dist < r_i + r_j, push apart
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(i + 1, N):
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-9:
                    dist = 1e-9 # Avoid division by zero
                
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    # Force proportional to overlap, inverse distance to avoid explosion
                    force_mag = overlap * 0.5 / dist
                    forces[i] += force_mag * vec / dist
                    forces[j] -= force_mag * vec / dist
        
        # Apply forces
        centers += forces
        centers = np.clip(centers, 0.05, 0.95) # Keep slightly away from edges

    # --- 3. Final LP to get precise radii for the best centers ---
    c_obj = np.ones(N) * -1
    A_ub = []
    b_ub = []
    dists = cdist(centers, centers)
    for i in range(N):
        for j in range(i + 1, N):
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    bounds_r = []
    for i in range(N):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        bounds_r.append((0, max_r))
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    radii = res.x
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
