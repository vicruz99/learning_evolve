import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def decode_params(params):
    """Decode optimization parameters into centers and radii.
    Uses parameterization x = r + u*(1-2r) to automatically satisfy boundary constraints."""
    r = params[0::3]
    u = params[1::3]
    v = params[2::3]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return x, y, r

def objective(params):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    _, _, r = decode_params(params)
    return -np.sum(r)

def constraints(params):
    """Constraint function: enforce non-overlapping circles.
    Returns array of values that must be >= 0: dist^2 - (r_i + r_j)^2"""
    x, y, r = decode_params(params)
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    sum_r = r[:, np.newaxis] + r[np.newaxis, :]
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    return (dist_sq - sum_r**2)[mask].flatten()

def run_packing():
    n = N_CIRCLES
    best_val = -np.inf
    best_x = None
    
    # Bounds: r in [0, 0.5], u in [0, 1], v in [0, 1]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 0.5), (0.0, 1.0), (0.0, 1.0)])
        
    cons = {'type': 'ineq', 'fun': constraints}
    
    np.random.seed(42)
    # Multiple restarts to escape local optima
    for trial in range(25):
        # Initialize with small feasible radii and random positions
        r0 = np.full(n, 0.09)
        u0 = np.random.rand(n)
        v0 = np.random.rand(n)
        
        p0 = np.zeros(3 * n)
        for i in range(n):
            p0[3*i] = r0[i]
            p0[3*i+1] = u0[i]
            p0[3*i+2] = v0[i]
            
        # Add perturbation to break symmetry and explore space
        p0 += np.random.randn(3*n) * 0.02
        p0[0::3] = np.clip(p0[0::3], 0.0, 0.5)
        p0[1::3] = np.clip(p0[1::3], 0.0, 1.0)
        p0[2::3] = np.clip(p0[2::3], 0.0, 1.0)
        
        res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 600, 'ftol': 1e-12, 'disp': False})
                       
        if not np.isnan(res.fun) and -res.fun > best_val:
            best_val = -res.fun
            best_x = res.x
            
    x, y, r = decode_params(best_x)
    centers = np.column_stack((x, y))
    return centers, r, best_val