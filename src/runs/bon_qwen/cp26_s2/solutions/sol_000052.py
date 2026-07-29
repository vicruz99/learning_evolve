# sol_000052 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 67b9141d) state=d9209377 sum of radii=2.568646 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constants to avoid closures
N_CIRCLES = 26
# Precompute pairwise indices for constraint evaluation
PAIR_INDICES = [(i, j) for i in range(N_CIRCLES) for j in range(i + 1, N_CIRCLES)]

def compute_objective(vars):
    """Negative sum of radii (minimization)"""
    radii = vars[2 * N_CIRCLES:]
    return -np.sum(radii)

def compute_constraints(vars):
    """Inequality constraints >= 0"""
    centers = vars[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = vars[2 * N_CIRCLES:]

    # Boundary constraints: dist to walls - radius >= 0
    con = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])

    # Pairwise non-overlap: dist^2 - (r_i + r_j)^2 >= 0
    # Vectorized computation for performance
    dx = centers[:, 0, np.newaxis] - centers[:, 0, np.newaxis].T
    dy = centers[:, 1, np.newaxis] - centers[:, 1, np.newaxis].T
    d2 = dx**2 + dy**2
    
    dr = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Extract upper triangle (unique pairs)
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    con = np.concatenate([con, d2[mask] - dr[mask]**2])
    
    return con

def run_packing():
    np.random.seed(42)
    
    best_centers = None
    best_radii = None
    best_obj = np.inf  # We minimize negative sum, so lower is better
    
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    constraint_dict = {'type': 'ineq', 'fun': compute_constraints}
    
    # Try multiple initializations to escape local minima
    for trial in range(3):
        # 1. Hexagonal-ish grid initialization
        xs = np.linspace(0.12, 0.88, 6)
        ys = np.linspace(0.12, 0.88, 5)
        grid = np.array([[x, y] for y in ys for x in xs])
        
        # Select first 26 points
        init_centers = grid[:N_CIRCLES].copy()
        
        # Add controlled noise to break symmetry
        init_centers += np.random.normal(0, 0.003, size=init_centers.shape)
        
        # Clip to valid range
        init_centers = np.clip(init_centers, 0.05, 0.95)
        
        # Initial radii: small enough to be valid, large enough to give optimizer headroom
        init_radii = np.full(N_CIRCLES, 0.045)
        
        x0 = np.concatenate([init_centers.flatten(), init_radii])
        
        # Run SLSQP optimization
        try:
            res = minimize(
                compute_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_dict,
                options={'maxiter': 1500, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
                best_radii = res.x[2 * N_CIRCLES:].copy()
        except Exception:
            continue

    # Fallback if optimization failed completely (highly unlikely with given init)
    if best_centers is None:
        xs = np.linspace(0.15, 0.85, 6)
        ys = np.linspace(0.15, 0.85, 5)
        grid = np.array([[x, y] for y in ys for x in xs])
        best_centers = grid[:N_CIRCLES]
        best_radii = np.full(N_CIRCLES, 0.05)

    # Post-processing: Ensure strict validity within tolerance
    centers = best_centers.copy()
    radii = best_radii.copy()
    
    # Iteratively shrink radii slightly if any constraint is violated due to numerical precision
    for _ in range(50):
        fixed = True
        # Check pairwise
        for i, j in PAIR_INDICES:
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-9:
                overlap = (radii[i] + radii[j] - dist) / 2.0 + 1e-6
                radii[i] -= overlap
                radii[j] -= overlap
                fixed = False
        # Check boundaries
        for i in range(N_CIRCLES):
            for dim in range(2):
                if centers[i, dim] - radii[i] < -1e-9:
                    radii[i] -= (-1e-9 - (centers[i, dim] - radii[i])) + 1e-6
                    fixed = False
                if centers[i, dim] + radii[i] > 1.0 + 1e-9:
                    radii[i] -= (centers[i, dim] + radii[i] - 1.0 - 1e-9) + 1e-6
                    fixed = False
        radii = np.maximum(radii, 0.0)
        if fixed:
            break
            
    # Ensure positive radii
    radii = np.maximum(radii, 1e-9)
    
    total_sum = np.sum(radii)
    return centers, radii, total_sum
