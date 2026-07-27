import numpy as np
from scipy.optimize import differential_evolution

def compute_objective(vars, N):
    """
    Computes the objective: minimize -sum(radii) + penalty for violations.
    vars: flattened array of shape (3*N,) containing [x1, y1, r1, x2, y2, r2, ...]
    """
    pts = vars.reshape(N, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    
    # Primary goal: maximize sum of radii
    sum_r = np.sum(r)
    
    # Quadratic boundary penalties
    pen = 0.0
    pen += np.sum(np.maximum(0, r - x)**2)
    pen += np.sum(np.maximum(0, x + r - 1)**2)
    pen += np.sum(np.maximum(0, r - y)**2)
    pen += np.sum(np.maximum(0, y + r - 1)**2)
    
    # Quadratic overlap penalties using vectorized broadcasting
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    r_sum = r[:, None] + r[None, :]
    overlap = np.maximum(0, r_sum - dist)
    np.fill_diagonal(overlap, 0.0)
    pen += np.sum(overlap**2)
    
    # Large penalty weight enforces validity strongly
    lam = 10000.0
    return -sum_r + lam * pen

def run_packing():
    N = 26
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    # Global optimization with Differential Evolution
    result = differential_evolution(
        compute_objective, 
        bounds, 
        args=(N,),
        maxiter=2000, 
        popsize=20,
        mutation=(0.5, 1.5), 
        recombination=0.7,
        seed=42, 
        tol=1e-7, 
        polish=True
    )
        
    best_vars = result.x
    pts = best_vars.reshape(N, 3)
    centers = pts[:, :2].copy()
    radii = np.maximum(pts[:, 2], 0.0)
    
    return centers, radii, np.sum(radii)