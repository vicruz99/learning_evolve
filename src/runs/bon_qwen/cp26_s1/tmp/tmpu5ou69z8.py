import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(vars, mu):
    """
    Computes the penalized objective: -sum(r) + mu * penalty
    vars: flattened array of [x0, y0, r0, x1, y1, r1, ...]
    mu: penalty weight
    """
    n = N_CIRCLES
    c = vars[:2*n].reshape(n, 2)
    r = vars[2::3]
    
    obj = -np.sum(r)
    pen = 0.0
    
    # Boundary penalties: ensure circles stay inside [0,1]x[0,1]
    # Violation is max(0, r - x), max(0, r - (1-x)), etc.
    pen += np.sum(np.maximum(0, r - c[:,0])**2)
    pen += np.sum(np.maximum(0, r - (1 - c[:,0]))**2)
    pen += np.sum(np.maximum(0, r - c[:,1])**2)
    pen += np.sum(np.maximum(0, r - (1 - c[:,1]))**2)
    
    # Overlap penalties: ensure dist >= r_i + r_j
    # Vectorized pairwise distance computation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    overlap = np.maximum(0, r_sum - dists)
    
    # Sum over all pairs and divide by 2 (matrix is symmetric, diagonal is 0)
    pen += 0.5 * np.sum(overlap**2)
    
    return obj + mu * pen

def objective_func(vars, mu):
    """Wrapper compatible with scipy.optimize.minimize signature"""
    return compute_objective(vars, mu)

def get_bounds():
    """Defines box constraints for x, y in [0,1] and r in [0, 0.5]"""
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return bounds

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii)
    """
    n = N_CIRCLES
    bounds = get_bounds()
    
    # 1. Hexagonal initialization
    # Places circles in a staggered grid approximating dense packing
    r_init = 0.08
    centers = np.zeros((n, 2))
    idx = 0
    row = 0
    while idx < n:
        col = 0
        while idx < n:
            x = 0.15 + col * 2 * r_init + (row % 2) * r_init
            y = 0.15 + row * np.sqrt(3) * r_init
            if x <= 0.9 and y <= 0.9:
                centers[idx] = [x, y]
                idx += 1
                col += 1
            else:
                break
        row += 1
        
    # Flatten to optimization variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.hstack([centers.ravel(), np.full(n, r_init)])
    
    # 2. Optimization with continuation method
    mu = 100.0
    vars_opt = x0
    
    # Iteratively increase penalty weight to tighten constraints
    for step in range(40):
        res = minimize(objective_func, vars_opt, args=(mu,), method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-14, 'gtol': 1e-12, 'maxiter': 5000, 'disp': False})
        vars_opt = res.x
        mu *= 1.4  # Geometric increase in penalty strength
        
    final_centers = vars_opt[:2*n].reshape(n, 2)
    final_radii = vars_opt[2::3]
    
    # 3. Safety margin
    # Shrink radii by a negligible factor to guarantee strict compliance with the 
    # validator's 1e-12 tolerance after floating-point operations.
    final_radii *= 0.9999999
    
    return final_centers, final_radii, float(np.sum(final_radii))