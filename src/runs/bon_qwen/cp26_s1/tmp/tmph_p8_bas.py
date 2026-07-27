import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n):
    """
    Objective function for circle packing.
    Minimizes negative sum of radii + penalty for overlaps and boundary violations.
    """
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Primary objective: maximize sum of radii
    val = -np.sum(r)
    
    # Pairwise overlap penalty
    # dist[i, j] = Euclidean distance between circle i and j
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Overlap condition: r_i + r_j > dist_ij
    overlap = r[:, np.newaxis] + r[np.newaxis, :] - dist
    np.fill_diagonal(overlap, 0)
    val += 1000 * np.sum(np.maximum(0, overlap)**2)
    
    # Boundary penalties: circle must stay within [0, 1]^2
    val += 1000 * np.sum(np.maximum(0, r - c[:, 0])**2)
    val += 1000 * np.sum(np.maximum(0, r - (1 - c[:, 0]))**2)
    val += 1000 * np.sum(np.maximum(0, r - c[:, 1])**2)
    val += 1000 * np.sum(np.maximum(0, r - (1 - c[:, 1]))**2)
    
    return val

def run_packing():
    np.random.seed(42)
    n = 26
    bounds = [(0, 1)] * 2*n + [(0, 0.5)] * n
    
    best_val = np.inf
    best_vars = None
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Uniform Grid (5x5 with one extra)
    c_grid = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < n:
                c_grid[idx] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                idx += 1
    starts.append(np.concatenate([c_grid.flatten(), np.full(n, 0.08)]))
    
    # 2. Random initializations
    for seed in range(8):
        np.random.seed(seed)
        c_rand = np.random.rand(n, 2) * 0.8 + 0.1
        starts.append(np.concatenate([c_rand.flatten(), np.full(n, 0.05)]))
        
    # 3. Hexagonal-like lattice
    c_hex = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < n:
                cx = (i + 0.5 + (j % 2) * 0.25) / 5.0
                cy = (j + 0.5) / 5.0
                c_hex[idx] = [cx, cy]
                idx += 1
    starts.append(np.concatenate([c_hex.flatten(), np.full(n, 0.08)]))

    # Optimize from each start
    for x0 in starts:
        res = minimize(compute_objective, x0, args=(n,), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-15})
        if res.fun < best_val:
            best_val = res.fun
            best_vars = res.x
            
    c_final = best_vars[:2*n].reshape(n, 2)
    r_final = best_vars[2*n:].copy()
    
    # Post-processing: deterministically fix any residual numerical violations
    for _ in range(100):
        improved = False
        
        # Fix pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((c_final[i] - c_final[j])**2))
                if d < r_final[i] + r_final[j] - 1e-10:
                    # Scale radii down just enough to separate them
                    scale = (d + 1e-9) / (r_final[i] + r_final[j])
                    r_final[i] *= np.sqrt(scale)
                    r_final[j] *= np.sqrt(scale)
                    improved = True
                    
        # Fix boundary violations
        for i in range(n):
            if c_final[i, 0] < r_final[i] - 1e-10:
                r_final[i] = c_final[i, 0]
                improved = True
            if c_final[i, 0] > 1 - r_final[i] + 1e-10:
                r_final[i] = 1 - c_final[i, 0]
                improved = True
            if c_final[i, 1] < r_final[i] - 1e-10:
                r_final[i] = c_final[i, 1]
                improved = True
            if c_final[i, 1] > 1 - r_final[i] + 1e-10:
                r_final[i] = 1 - c_final[i, 1]
                improved = True
                
        if not improved:
            break
            
    return c_final, r_final, np.sum(r_final)