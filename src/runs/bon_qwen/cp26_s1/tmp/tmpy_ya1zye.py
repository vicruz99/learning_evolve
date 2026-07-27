import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, N):
    """Objective function: minimize negative sum of radii"""
    rs = vars[2*N:]
    return -np.sum(rs)

def compute_constraints(vars, N):
    """Constraint function: returns array of constraint values >= 0"""
    cs = vars[:2*N].reshape(N, 2)
    rs = vars[2*N:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    b1 = cs[:, 0] - rs
    b2 = 1 - cs[:, 0] - rs
    b3 = cs[:, 1] - rs
    b4 = 1 - cs[:, 1] - rs
    boundaries = np.concatenate([b1, b2, b3, b4])
    
    # Overlap constraints: dist_sq >= (r_i + r_j)^2
    # Vectorized pairwise distance squared
    dx = cs[:, 0][:, None] - cs[:, 0][None, :]
    dy = cs[:, 1][:, None] - cs[:, 1][None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = rs[:, None] + rs[None, :]
    
    # Upper triangle mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlaps = (dist_sq - r_sum**2)[mask]
    
    return np.concatenate([boundaries, overlaps])

def run_packing():
    N = 26
    
    # Initialize with a hexagonal lattice pattern
    # We aim to place points densely. 5 rows, 6 columns approximates 30 points.
    # We'll select the first 26.
    cols = 6
    rows = 5
    xs = np.linspace(0.12, 0.88, cols)
    ys = np.linspace(0.12, 0.88, rows)
    
    pts = []
    for r_idx in range(rows):
        offset = (r_idx % 2) * (xs[1] - xs[0]) / 2.0
        for c_idx in range(cols):
            if len(pts) >= N:
                break
            pts.append([xs[c_idx] + offset, ys[r_idx]])
        if len(pts) >= N:
            break
            
    centers_init = np.array(pts)
    
    # Add small perturbations to break symmetry
    rng = np.random.default_rng(42)
    centers_init += rng.normal(0, 0.015, centers_init.shape)
    centers_init = np.clip(centers_init, 0.05, 0.95)
    
    # Build initial variable vector: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = 0.08  # Initial radius guess
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    
    # Setup constraints
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (N,)}
    
    # Run optimization
    res = minimize(
        compute_objective, 
        x0, 
        args=(N,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False}
    )
    
    # Extract results
    centers = res.x[:2*N].reshape(N, 2)
    radii = res.x[2*N:]
    
    # Ensure radii are non-negative (safety clamp)
    radii = np.maximum(radii, 0.0)
    
    # Calculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii