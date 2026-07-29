# sol_000079 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3e28a2dc) state=3728655e sum of radii=2.085231 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def boundary_constraint(x, n):
    """Enforces that circles remain inside the unit square [0,1]x[0,1]."""
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    return np.concatenate([
        cx - cr,
        1.0 - cx - cr,
        cy - cr,
        1.0 - cy - cr
    ])

def overlap_constraint(x, n):
    """Enforces that no two circles overlap: distance >= sum of radii."""
    cx = x[0::3]
    cy = x[1::3]
    cr = x[2::3]
    
    # Vectorized pairwise distances
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Vectorized pairwise radius sums
    r_sums = cr[:, np.newaxis] + cr[np.newaxis, :]
    
    # Extract upper triangle to avoid duplicates and self-pairs
    idx = np.triu_indices(n, k=1)
    return dists[idx] - r_sums[idx]

def objective(x):
    """Objective: minimize negative sum of radii (equivalent to maximizing sum)."""
    return -np.sum(x[2::3])

def run_packing():
    np.random.seed(42)
    n = 26
    best_x = None
    best_obj = np.inf
    
    # Run multiple trials with perturbed initial positions to escape local minima
    for trial in range(3):
        centers = np.zeros((n, 2))
        radii_init = np.full(n, 0.06)
        
        idx = 0
        # Generate a staggered hexagonal-ish grid
        for r in range(6):
            for c in range(6):
                if idx >= n:
                    break
                x = (c + 0.5 + (0.5 if r % 2 == 0 else 0.0)) * (1.0 / 6.5)
                y = (r + 0.5) * (1.0 / 6.5)
                # Add small random perturbation
                x += np.random.uniform(-0.015, 0.015)
                y += np.random.uniform(-0.015, 0.015)
                # Ensure initial positions are safely inside bounds
                x = np.clip(x, 0.12, 0.88)
                y = np.clip(y, 0.12, 0.88)
                centers[idx] = [x, y]
                idx += 1
            if idx >= n:
                break
                
        # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3 * i] = centers[i, 0]
            x0[3 * i + 1] = centers[i, 1]
            x0[3 * i + 2] = radii_init[i]
            
        # Bounds: coordinates in [0,1], radii in [0, 0.5]
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        
        constraints = [
            {'type': 'ineq', 'fun': boundary_constraint, 'args': (n,)},
            {'type': 'ineq', 'fun': overlap_constraint, 'args': (n,)}
        ]
        
        # Optimize
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 5000, 'ftol': 1e-10})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x.copy()
            
    # Extract results
    final_centers = best_x.reshape(-1, 3)[:, :2]
    final_radii = best_x.reshape(-1, 3)[:, 2]
    
    # Ensure strict non-negativity (should already hold due to bounds)
    final_radii = np.clip(final_radii, 0, None)
    
    return final_centers, final_radii, -best_obj
