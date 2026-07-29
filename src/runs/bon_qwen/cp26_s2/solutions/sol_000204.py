# sol_000204 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cda7e5e4) state=46254054 sum of radii=1.857143 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a physics-based repulsion simulation combined with Linear Programming
    for radius optimization.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # 1. Initialization: Dense 6x5 grid pattern
    # We use 6 columns and 5 rows, taking the first 26 points.
    # This ensures good coverage of the square.
    x_spacing = 1.0 / (6 + 1)  # 1/7 approx 0.142
    y_spacing = 1.0 / (5 + 1)  # 1/6 approx 0.166
    
    idx = 0
    # Fill grid
    for r in range(5):
        for c in range(6):
            if idx < n:
                # Offset slightly to center within grid cells
                centers[idx, 0] = (c + 1) * x_spacing
                centers[idx, 1] = (r + 1) * y_spacing
                idx += 1

    # 2. Optimization Loop
    # Physics parameters
    learning_rate = 0.005
    force_scaling = 10.0
    epsilon = 1e-7
    
    for step in range(300):
        # --- Step A: Optimize Radii using Linear Programming ---
        # Maximize sum(r_i) subject to:
        # r_i + r_j <= dist(i, j)
        # r_i <= boundary_distance(i)
        # r_i >= 0
        
        c_obj = -np.ones(n) # Minimize -sum(r) is maximize sum(r)
        A_ub = []
        b_ub = []
        
        # Pairwise distance constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(dist)
        
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            bounds_val = [x, 1 - x, y, 1 - y]
            for b_val in bounds_val:
                row = np.zeros(n)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(b_val)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds r_i >= 0
        bounds = [(0, None) for _ in range(n)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                radii = res.x
            else:
                # Fallback to simple projection if LP fails (rare)
                radii = np.min(np.vstack([
                    np.linalg.norm(centers[i] - centers[j]) for j in range(n)
                ]), axis=0) # Very rough estimate
                radii = np.clip(radii, 0, None)
        except Exception:
            radii = np.zeros(n)

        # --- Step B: Update Centers using Repulsion Forces ---
        forces = np.zeros_like(centers)
        
        # Pairwise Repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < epsilon:
                    continue
                
                # Required distance for non-overlap
                req_dist = radii[i] + radii[j]
                
                # If they are touching or overlapping, push apart
                if req_dist > dist - epsilon:
                    # Force proportional to how much they exceed the distance
                    # or simply push if touching to allow expansion
                    overlap = req_dist - dist
                    if overlap < 0: overlap = 0
                    
                    # Unit vector direction
                    dir_vec = diff / dist
                    
                    # Force magnitude
                    f_mag = overlap * force_scaling
                    forces[i] += dir_vec * f_mag
                    forces[j] -= dir_vec * f_mag
        
        # Boundary Repulsion (if limited by wall, push inwards)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left Wall (x >= r) -> if x-r ~ 0, push x up
            if x - r < epsilon:
                forces[i, 0] += (epsilon - (x - r)) * force_scaling
            # Right Wall (x + r <= 1) -> if 1-x-r ~ 0, push x down
            if (1 - x) - r < epsilon:
                forces[i, 0] -= (epsilon - ((1 - x) - r)) * force_scaling
            # Bottom Wall (y >= r) -> if y-r ~ 0, push y up
            if y - r < epsilon:
                forces[i, 1] += (epsilon - (y - r)) * force_scaling
            # Top Wall (y + r <= 1) -> if 1-y-r ~ 0, push y down
            if (1 - y) - r < epsilon:
                forces[i, 1] -= (epsilon - ((1 - y) - r)) * force_scaling
                
        # Apply forces
        centers += forces * learning_rate
        
        # Clip centers to valid range [0, 1]
        # Though forces should keep them inside, clipping ensures safety
        centers = np.clip(centers, 0, 1)

    # 3. Final Radius Optimization
    # Re-run LP one last time to ensure radii are optimal for final centers
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    for i in range(n):
        x, y = centers[i]
        bounds_val = [x, 1 - x, y, 1 - y]
        for b_val in bounds_val:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b_val)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None) for _ in range(n)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        final_radii = res.x
    else:
        final_radii = np.zeros(n)
        
    sum_radii = np.sum(final_radii)
    
    return centers, final_radii, sum_radii
