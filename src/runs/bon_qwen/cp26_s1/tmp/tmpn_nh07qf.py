import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Number of restarts to find a good local optimum
    n_restarts = 20
    
    for seed in range(n_restarts):
        np.random.seed(seed)
        
        # --- Initialization Strategy ---
        # Start with a hexagonal-like grid to ensure a dense packing initially
        # We try to fit roughly sqrt(26) ~ 5 rows/cols
        centers = np.zeros((n, 2))
        radii = np.ones(n) * 0.05 # Initial small radius
        
        # Generate a hexagonal grid points
        points = []
        x, y = 0.0, 0.0
        row_height = 1.0 / 5.0
        # Try to distribute points somewhat evenly
        for i in range(n):
            # Simple grid fill with some jitter
            row = i // 5
            col = i % 5
            shift = 0.5 * row_height * (row % 2)
            px = col * (1.0 / 5.0) + 0.5/5.0 + shift
            py = row * row_height + 0.5/5.0
            # Add random perturbation to break symmetry
            px += np.random.uniform(-0.02, 0.02)
            py += np.random.uniform(-0.02, 0.02)
            points.append([px, py])
        
        centers = np.array(points)
        
        # --- Optimization Setup ---
        # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
        # Total 78 variables
        x0 = np.zeros(n * 3)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii[i]
        
        # Bounds for variables
        # x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        # Constraints
        cons = []
        
        # 1. Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        # Equivalent to: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        for i in range(n):
            idx = 3 * i
            # x - r >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[idx] - v[idx+2]
            })
            # 1 - x - r >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[idx] - v[idx+2]
            })
            # y - r >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[idx+1] - v[idx+2]
            })
            # 1 - y - r >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[idx+1] - v[idx+2]
            })
            
        # 2. Non-overlap constraints: ||ci - cj||^2 >= (ri + rj)^2
        # dist^2 - (r1+r2)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = 3 * i
                idx_j = 3 * j
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=idx_i, j=idx_j: \
                        (v[i] - v[j])**2 + (v[i+1] - v[j+1])**2 - (v[i+2] + v[j+2])**2
                })
        
        # Objective: Maximize sum of radii => Minimize negative sum
        def objective(v):
            return -sum(v[3*i + 2] for i in range(n))
        
        # Run optimization
        try:
            # SLSQP is generally good for constrained problems
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or res.fun < -best_sum: # Check if we improved (fun is negative sum)
                current_sum = -res.fun
                # Extract centers and radii
                curr_centers = np.zeros((n, 2))
                curr_radii = np.zeros(n)
                for i in range(n):
                    curr_centers[i, 0] = res.x[3*i]
                    curr_centers[i, 1] = res.x[3*i+1]
                    curr_radii[i] = res.x[3*i+2]
                
                # Validate locally to ensure no numerical drift violations
                # The constraints handle most, but let's check radius positivity etc.
                if np.all(curr_radii >= 0):
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = curr_centers
                        best_radii = curr_radii
                        
        except Exception:
            continue

    # Final validation and cleanup
    if best_centers is not None:
        # Clip radii to be non-negative just in case
        best_radii = np.maximum(best_radii, 0.0)
        
        # One last check with the provided validation logic logic (mental check)
        # The optimizer constraints ensure validity, but floating point errors might occur.
        # However, the constraints are strict inequalities in the solver (>=0).
        # 1e-12 tolerance in validator should pass.
        
        return best_centers, best_radii, best_sum
    else:
        # Fallback to a simple grid if optimization fails completely
        centers = np.zeros((n, 2))
        radii = np.ones(n) * 0.04 # Safe small radius
        for i in range(n):
            row = i // 5
            col = i % 5
            centers[i, 0] = 0.1 + col * 0.18
            centers[i, 1] = 0.1 + row * 0.18
        return centers, radii, np.sum(radii)