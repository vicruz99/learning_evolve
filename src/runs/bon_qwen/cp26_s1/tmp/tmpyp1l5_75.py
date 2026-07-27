import numpy as np
from scipy.optimize import minimize

def objective(v):
    """Objective function: maximize sum of radii => minimize negative sum."""
    n = len(v) // 3
    return -np.sum(v[2*n:])

def constraint_overlap(v):
    """Non-overlap constraints: dist(c_i, c_j) >= r_i + r_j"""
    n = len(v) // 3
    centers = v[:2*n].reshape(n, 2)
    radii = v[2*n:]
    
    # Compute pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Lower triangle indices
    i, j = np.tril_indices(n, -1)
    return dists[i, j] - radii[i] - radii[j]

def constraint_boundary(v):
    """Boundary constraints: circles must be inside [0,1]x[0,1]"""
    n = len(v) // 3
    centers = v[:2*n].reshape(n, 2)
    radii = v[2*n:]
    
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    return np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])

def run_packing():
    n = 26
    
    # 1. Initial Guess: Hexagonal Grid
    # Hexagonal packing is optimal for density.
    # Estimate initial radius to ensure feasibility
    r_init = 0.08
    centers = []
    
    # Generate hexagonal lattice points
    dy = np.sqrt(3) * r_init
    dx = 2.0 * r_init
    y = 0.5 + r_init
    row = 0
    while len(centers) < n:
        x = 0.5 + r_init + (row % 2) * dx / 2
        while x < 1.0 - r_init and len(centers) < n:
            centers.append([x, y])
            x += dx
        y += dy
        row += 1
        
    centers = np.array(centers[:n])
    
    # Prepare initial vector
    radii_init = np.ones(n) * r_init
    x0 = np.concatenate([centers.flatten(), radii_init])
    
    # Bounds for variables: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # Constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_overlap},
        {'type': 'ineq', 'fun': constraint_boundary}
    ]
    
    # 2. Optimization
    # SLSQP handles non-linear constraints well
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False})
                   
    centers_opt = res.x[:2*n].reshape(n, 2)
    radii_opt = res.x[2*n:]
    
    # 3. Post-processing to ensure strict feasibility
    # Enforce boundary constraints
    for i in range(n):
        x, y = centers_opt[i]
        r = radii_opt[i]
        r = min(r, x, 1.0 - x, y, 1.0 - y)
        radii_opt[i] = max(r, 0.0)
        
    # Enforce non-overlap constraints
    # If circles overlap, scale down their radii proportionally to just touch
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers_opt[i] - centers_opt[j])
            sum_r = radii_opt[i] + radii_opt[j]
            if sum_r > dist + 1e-12:
                scale = dist / sum_r if sum_r > 1e-12 else 0.0
                radii_opt[i] *= scale
                radii_opt[j] *= scale
                
    total_radius = float(np.sum(radii_opt))
    return centers_opt, radii_opt, total_radius