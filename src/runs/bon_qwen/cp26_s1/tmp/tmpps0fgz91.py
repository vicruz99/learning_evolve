import numpy as np
from scipy.optimize import minimize

def compute_loss(vars, n, lam):
    """
    Computes the optimization loss: maximizes radius r while penalizing overlaps and boundary violations.
    vars: array of shape (2*n + 1) containing x, y for each circle, then r.
    n: number of circles.
    lam: penalty weight.
    """
    r = vars[-1]
    centers = vars[:-1].reshape(n, 2)
    
    # Wall violations: circles must be inside [0, 1]
    wall_viol = (np.maximum(0, r - centers[:,0])**2 + 
                 np.maximum(0, centers[:,0] - (1.0 - r))**2 + 
                 np.maximum(0, r - centers[:,1])**2 + 
                 np.maximum(0, centers[:,1] - (1.0 - r))**2)
                 
    # Pairwise overlap violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    pair_viol = np.maximum(0, 2.0 * r - dist)
    
    # Objective: maximize r (minimize -r) + penalties
    return -r + lam * (np.sum(wall_viol) + np.sum(pair_viol**2))

def optimize_from_init(init_vars, n, lam):
    """Runs BFGS optimization from a given initialization."""
    res = minimize(compute_loss, init_vars, args=(n, lam), method='BFGS', 
                   options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-12})
    return res

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns: (centers, radii, sum_radii)
    """
    np.random.seed(42)
    n = 26
    lam = 800.0  # High penalty to enforce constraints strictly during optimization
    best_res = None
    best_loss = np.inf
    
    # Generate diverse initial configurations to escape local minima
    inits = []
    
    # 1. Structured grid (5x5 + center)
    grid = np.linspace(0.12, 0.88, 5)
    c1 = np.array([(x, y) for x in grid for y in grid])
    c1 = np.vstack([c1, [0.5, 0.5]])  # 26th circle
    inits.append(np.concatenate([c1.flatten(), [0.095]]))
    
    # 2. Random uniform inside safe margin
    c2 = np.random.uniform(0.15, 0.85, (n, 2))
    inits.append(np.concatenate([c2.flatten(), [0.095]]))
    
    # 3. Perturbed grid
    c3 = c1 + np.random.uniform(-0.025, 0.025, (n, 2))
    c3 = np.clip(c3, 0.05, 0.95)
    inits.append(np.concatenate([c3.flatten(), [0.095]]))
    
    # Optimize from each start
    for init_vars in inits:
        res = optimize_from_init(init_vars, n, lam)
        if res.fun < best_loss:
            best_loss = res.fun
            best_res = res
            
    # Extract best solution
    vars_opt = best_res.x
    r_opt = vars_opt[-1]
    centers_opt = vars_opt[:-1].reshape(n, 2)
    
    # Post-processing: Enforce strict validity by computing exact limits
    # Pairwise distances
    dists = np.sqrt(np.sum((centers_opt[:, None, :] - centers_opt[None, :, :])**2, axis=2) + 1e-12)
    min_pair_dist = np.min(dists[np.triu_indices(n, k=1)])
    r_pair = min_pair_dist / 2.0
    
    # Boundary distances
    r_wall = np.min([np.min(centers_opt[:,0]), np.min(centers_opt[:,1]), 
                     np.max(1.0 - centers_opt[:,0]), np.max(1.0 - centers_opt[:,1])])
    
    # Choose smallest valid radius, apply tiny safety margin for numerical stability
    final_r = max(1e-6, min(r_opt, r_pair, r_wall) - 1e-9)
    radii = np.full(n, final_r)
    
    return centers_opt, radii, np.sum(radii)