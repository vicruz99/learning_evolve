# sol_000343 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 776f37f0) state=f7b89fe3 sum of radii=2.600498 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def boundary_constraints(x):
    """Enforce circles stay within the unit square."""
    n = 26
    out = np.empty(4 * n)
    idx = 0
    for i in range(n):
        xi = x[3*i]
        yi = x[3*i+1]
        ri = x[3*i+2]
        out[idx] = xi - ri          # x >= r
        out[idx+1] = 1.0 - xi - ri  # x <= 1-r
        out[idx+2] = yi - ri        # y >= r
        out[idx+3] = 1.0 - yi - ri  # y <= 1-r
        idx += 4
    return out

def pairwise_constraints(x):
    """Enforce non-overlapping circles."""
    n = 26
    m = n * (n - 1) // 2
    out = np.empty(m)
    k = 0
    for i in range(n):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        for j in range(i + 1, n):
            xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
            dx, dy = xi - xj, yi - yj
            out[k] = dx*dx + dy*dy - (ri + rj)**2
            k += 1
    return out

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Initial hexagonal configuration
    r0 = 0.08
    centers = []
    row_lengths = [6, 5, 6, 5, 4]
    y = r0
    row_idx = 0
    h = np.sqrt(3) * r0
    w = 2 * r0
    
    for length in row_lengths:
        total_w = (length - 1) * w
        x_start = (1 - total_w) / 2
        for k in range(length):
            x = x_start + k * w
            if row_idx % 2 == 1:
                x += w / 2
            centers.append([x, y])
        y += h
        row_idx += 1
        
    centers = np.array(centers)
    radii = np.full(n, r0)
    # Deterministic perturbation to break symmetry and encourage unequal sizes if beneficial
    radii += np.linspace(-0.005, 0.005, n)
    radii = np.maximum(radii, 0.01)
    
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    cons = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': pairwise_constraints}
    ]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
    
    if res.success:
        final_x = res.x
    else:
        final_x = x0
        
    final_centers = final_x.reshape(n, 3)[:, :2]
    final_radii = final_x[2::3]
    final_radii = np.maximum(final_radii, 0.0)
    
    sum_r = np.sum(final_radii)
    return final_centers, final_radii, sum_r
