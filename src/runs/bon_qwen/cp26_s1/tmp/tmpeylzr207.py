import numpy as np
from scipy.optimize import minimize
from functools import partial

def objective(vars):
    """Objective function: minimize negative sum of radii"""
    return -np.sum(vars[52:])

def pair_constraint(vars, i, j):
    """Non-overlap constraint: squared distance >= (r_i + r_j)^2"""
    xi, yi = 3*i, 3*i+1
    xj, yj = 3*j, 3*j+1
    ri, rj = 52+i, 52+j
    dx = vars[xi] - vars[xj]
    dy = vars[yi] - vars[yj]
    return dx*dx + dy*dy - (vars[ri] + vars[rj])**2

def wall_constraint(vars, i, dim, is_max):
    """Boundary constraint: 0 <= coord +/- r <= 1"""
    idx = 3*i + dim
    r_idx = 52 + i
    if is_max:
        return 1.0 - vars[idx] - vars[r_idx]
    else:
        return vars[idx] - vars[r_idx]

def run_packing():
    n = 26
    # Initial hexagonal packing configuration
    r_start = 0.075
    dy_hex = np.sqrt(3) * r_start
    centers = []
    row_counts = [6, 5, 6, 5, 4]
    
    for i, count in enumerate(row_counts):
        y = r_start + i * dy_hex
        row_width = (count - 1) * 2 * r_start
        x_start = (1 - row_width) / 2
        for j in range(count):
            centers.append([x_start + j * 2 * r_start, y])

    centers = np.array(centers)
    radii = np.full(n, r_start)
    x0 = np.concatenate([centers.ravel(), radii])
    
    # Small perturbation to break symmetry and aid convergence
    np.random.seed(42)
    x0 += np.random.uniform(-1e-5, 1e-5, size=x0.shape)

    # Build constraints
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': partial(pair_constraint, i=i, j=j)})
            
    for i in range(n):
        for dim in [0, 1]:
            for is_max in [False, True]:
                cons.append({'type': 'ineq', 'fun': partial(wall_constraint, i=i, dim=dim, is_max=is_max)})

    # Variable bounds
    bounds = [(0.0, 1.0)] * 52 + [(0.0, 0.5)] * 26

    # Optimize
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})

    # Extract and clean results
    final_centers = res.x[:52].reshape((n, 2))
    final_radii = res.x[52:]

    # Ensure strict feasibility within numerical tolerance
    final_radii = np.maximum(final_radii, 0.0)
    final_centers = np.clip(final_centers, 0.0, 1.0)

    return final_centers, final_radii, float(np.sum(final_radii))