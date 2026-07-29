# sol_000347 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9cbd6fd8) state=0763f360 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Uses a hexagonal grid initialization followed by force-directed relaxation 
    and numerical optimization.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # --- 1. Hexagonal Initialization ---
    # Pattern: 6, 5, 6, 5, 4 circles (Total 26)
    # This staggered arrangement maximizes packing density.
    r_init = 0.07
    idx = 0
    row_counts = [6, 5, 6, 5, 4]
    
    for row_idx, count in enumerate(row_counts):
        # Hexagonal vertical step
        y = r_init + row_idx * (r_init * np.sqrt(3))
        
        # Stagger x-offset for alternate rows
        x_start = r_init
        if row_idx % 2 == 1:
            x_start = 2 * r_init
        
        for col_idx in range(count):
            x = x_start + col_idx * (2 * r_init)
            centers[idx] = [x, y]
            idx += 1
            
    # --- 2. Force-Directed Relaxation (Growing) ---
    # Iteratively grow circles and let them push each other apart
    r_current = 0.07
    steps = 800
    
    # Use numpy for performance
    c = centers.copy()
    
    for step in range(steps):
        # Slowly increase target radius
        target_r = r_current + 0.0003
        
        # Forces
        forces = np.zeros_like(c)
        
        # Boundary repulsion
        for i in range(n):
            if c[i, 0] < target_r: forces[i, 0] += (target_r - c[i, 0]) * 100
            if c[i, 0] > 1.0 - target_r: forces[i, 0] -= (c[i, 0] - (1.0 - target_r)) * 100
            if c[i, 1] < target_r: forces[i, 1] += (target_r - c[i, 1]) * 100
            if c[i, 1] > 1.0 - target_r: forces[i, 1] -= (c[i, 1] - (1.0 - target_r)) * 100
            
        # Inter-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                min_dist = 2 * target_r
                if dist < min_dist and dist > 1e-8:
                    overlap = min_dist - dist
                    repulsion = overlap * 50.0 / dist
                    f_x = dx * repulsion
                    f_y = dy * repulsion
                    forces[i, 0] += f_x
                    forces[i, 1] += f_y
                    forces[j, 0] -= f_x
                    forces[j, 1] -= f_y
        
        # Update positions with a damping factor to prevent explosion
        learning_rate = 0.05
        c += learning_rate * forces
        
        # Hard clip to ensure validity during simulation
        c[:, 0] = np.clip(c[:, 0], target_r, 1.0 - target_r)
        c[:, 1] = np.clip(c[:, 1], target_r, 1.0 - target_r)

    # --- 3. Refinement ---
    # Once the simulation stabilizes, use scipy to find the absolute optimal 
    # center coordinates for the best possible minimum separation.
    
    def objective(params):
        # params is 52 values (26 x, 26 y)
        pts = params.reshape((n, 2))
        min_d = 1.0
        
        # Boundary distances
        min_d = min(min_d, np.min(pts[:, 0]), np.min(1 - pts[:, 0]),
                    np.min(pts[:, 1]), np.min(1 - pts[:, 1]))
        
        # Inter-point distances
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d < min_d:
                    min_d = d
                    
        return -min_d # Minimize negative distance (Maximize distance)

    # Convert centers to flat array for optimization
    x0 = c.flatten()
    
    # Bounds to keep inside the square (approximate)
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Run optimization
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 500, 'ftol': 1e-9})
    
    final_centers = res.x.reshape((n, 2))
    
    # --- 4. Final Radius Calculation ---
    # Determine the largest uniform radius possible for these centers
    min_sep = 1.0
    
    # Check boundaries
    for i in range(n):
        r_bound = min(final_centers[i, 0], 1 - final_centers[i, 0],
                      final_centers[i, 1], 1 - final_centers[i, 1])
        if r_bound < min_sep:
            min_sep = r_bound
            
    # Check circle distances
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
            half_dist = dist / 2.0
            if half_dist < min_sep:
                min_sep = half_dist
                
    final_radii = np.full(n, min_sep)
    total_sum = np.sum(final_radii)
    
    # Add a tiny safety margin to handle floating point noise
    final_radii *= 0.9999
    
    return final_centers, final_radii, total_sum
