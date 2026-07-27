import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def compute_objective(v, mu, n):
    """
    Computes the objective: negative sum of radii plus penalty for overlaps and boundary violations.
    """
    cx = v[0::3]
    cy = v[1::3]
    r = v[2::3]
    
    pen = 0.0
    # Boundary penalties: circles must satisfy r <= x <= 1-r and r <= y <= 1-r
    pen += np.sum(np.maximum(r - cx, 0.0)**2)
    pen += np.sum(np.maximum(r + cx - 1.0, 0.0)**2)
    pen += np.sum(np.maximum(r - cy, 0.0)**2)
    pen += np.sum(np.maximum(r + cy - 1.0, 0.0)**2)
    
    # Overlap penalties: distance between centers must be >= r_i + r_j
    pts = np.column_stack((cx, cy))
    dist = cdist(pts, pts)
    r_sum = r[:, None] + r[None, :]
    overlap = r_sum - dist
    np.fill_diagonal(overlap, -1.0) # Ignore self-distances
    pen += np.sum(np.maximum(overlap, 0.0)**2)
    
    return -np.sum(r) + mu * pen

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Try multiple seeds to find the global optimum
    for seed in range(8):
        np.random.seed(seed)
        
        # Hexagonal initialization
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.06)
        
        idx = 0
        y = 0.1
        while idx < n and y < 0.9:
            x = 0.1
            while idx < n and x < 0.9:
                centers[idx] = [x, y]
                idx += 1
                x += 0.13
            y += 0.13
            
        # Fill any remaining slots
        while idx < n:
            centers[idx] = [np.random.rand(), np.random.rand()]
            idx += 1
            
        # Flatten to optimization vector
        v0 = np.zeros(n*3)
        for i in range(n):
            v0[3*i] = centers[i, 0]
            v0[3*i+1] = centers[i, 1]
            v0[3*i+2] = radii[i]
            
        # Stage 1: Positional optimization with annealing penalty
        mu = 20.0
        for _ in range(8):
            res = minimize(compute_objective, v0, args=(mu, n), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9})
            v0 = res.x
            mu *= 2.0
            
        cx = v0[0::3]
        cy = v0[1::3]
        r = v0[2::3]
        
        # Stage 2: Grow and Relax
        # Iteratively increase radii and resolve overlaps to push against constraints
        for _ in range(40):
            r[:] *= 1.0015
            v0[2::3] = r
            res = minimize(compute_objective, v0, args=(mu, n), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 300})
            v0 = res.x
            if res.success is False and res.nit == 0:
                break
                
        cx = v0[0::3]
        cy = v0[1::3]
        r = v0[2::3]
        
        # Ensure radii are strictly positive
        r = np.maximum(r, 1e-6)
        
        cur_sum = np.sum(r)
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = np.column_stack((cx, cy))
            best_radii = r.copy()
            
    return best_centers, best_radii, best_sum