import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Initialize with a perturbed 5x5 grid + 1 center circle
    grid = np.linspace(0, 1, 5)
    u0 = np.tile(grid, 5)
    v0 = np.repeat(grid, 5)
    u0 = np.append(u0, 0.5)
    v0 = np.append(v0, 0.5)
    
    # Add small random jitter to break symmetry and escape local minima
    np.random.seed(42)
    u0 += np.random.uniform(-0.02, 0.02, size=n)
    v0 += np.random.uniform(-0.02, 0.02, size=n)
    u0 = np.clip(u0, 0.0, 1.0)
    v0 = np.clip(v0, 0.0, 1.0)
    
    r0 = np.full(n, 0.09)
    
    # Pack variables: u0, v0, r0, u1, v1, r1, ...
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3 * i] = u0[i]
        x0[3 * i + 1] = v0[i]
        x0[3 * i + 2] = r0[i]
        
    # Bounds: u, v in [0, 1], r in [1e-5, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.5)] * n
    
    def get_centers_radii(vars):
        us = vars[0::3]
        vs = vars[1::3]
        rs = vars[2::3]
        # Map normalized coords to actual coords ensuring boundary constraints
        xs = rs + (1.0 - 2.0 * rs) * us
        ys = rs + (1.0 - 2.0 * rs) * vs
        return xs, ys, rs

    def objective(vars):
        return -np.sum(vars[2::3])
        
    def constraints(vars):
        xs, ys, rs = get_centers_radii(vars)
        # Vectorized pairwise distances
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        # Constraint: dist >= r_i + r_j  =>  dist - (r_i + r_j) >= 0
        cons = dist - (rs[:, None] + rs[None, :])
        # Keep only upper triangle to avoid duplicates
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return cons[mask].ravel()
        
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    # Run optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons_dict, 
                   options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
    
    # Extract results
    xs, ys, rs = get_centers_radii(res.x)
    centers = np.column_stack((xs, ys))
    radii = rs
    total_sum = np.sum(radii)
    
    return centers, radii, total_sum