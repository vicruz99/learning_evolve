import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute indices for pairwise constraints to speed up evaluation
_PAIR_INDICES = np.triu_indices(N, k=1)

def objective(params):
    """Objective: minimize negative sum of radii (i.e., maximize sum of radii)"""
    radii = params[2::3]
    return -np.sum(radii)

def constraints(params):
    """Returns array of constraint values that must be >= 0"""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Boundary constraints: 0 <= x-r, x+r <= 1, 0 <= y-r, y+r <= 1
    c_bounds = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    ix, iy = _PAIR_INDICES
    dx = x[ix] - x[iy]
    dy = y[ix] - y[iy]
    dr = r[ix] + r[iy]
    c_overlap = dx**2 + dy**2 - dr**2
    
    return np.concatenate([c_bounds, c_overlap])

def get_initial_params():
    """Generate a hexagonal lattice initialization"""
    params = np.zeros(3 * N)
    row_counts = [6, 5, 6, 5, 4]
    idx = 0
    y = 0.15
    y_step = 0.17
    
    for row_i, count in enumerate(row_counts):
        x_start = 0.12 + (0.08 if row_i % 2 == 1 else 0)
        x_step = 0.16
        for _ in range(count):
            params[3 * idx] = x_start
            params[3 * idx + 1] = y
            params[3 * idx + 2] = 0.06  # Conservative initial radius
            idx += 1
            x_start += x_step
        y += y_step
        
    return params

def run_packing():
    """Run the optimization and return the packing configuration"""
    x0 = get_initial_params()
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    # SLSQP is effective for this smooth constrained optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={'maxiter': 3000, 'ftol': 1e-11}
    )
    
    # Extract and format results
    centers = res.x.reshape(N, 3)[:, :2]
    radii = res.x.reshape(N, 3)[:, 2]
    
    # Ensure non-negative radii due to numerical tolerance
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, np.sum(radii)