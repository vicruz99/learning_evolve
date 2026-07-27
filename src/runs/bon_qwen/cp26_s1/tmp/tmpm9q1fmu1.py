import numpy as np
from scipy.optimize import minimize
import math

def objective(v):
    """Maximize sum of radii (minimize negative sum)."""
    n = 26
    return -np.sum(v[2*n:])

def constraint_func(v):
    """
    Returns an array of constraint values.
    All values must be >= 0 for feasibility.
    Includes boundary and non-overlap constraints.
    """
    n = 26
    # Boundary constraints: 4*n
    bounds = np.zeros(4*n)
    for i in range(n):
        x_idx = 2*i
        y_idx = 2*i + 1
        r_idx = 2*n + i
        bounds[4*i] = v[x_idx] - v[r_idx]          # x - r >= 0
        bounds[4*i+1] = 1.0 - v[x_idx] - v[r_idx]  # 1 - x - r >= 0
        bounds[4*i+2] = v[y_idx] - v[r_idx]        # y - r >= 0
        bounds[4*i+3] = 1.0 - v[y_idx] - v[r_idx]  # 1 - y - r >= 0
        
    # Overlap constraints: n*(n-1)/2
    n_pairs = n * (n - 1) // 2
    overlaps = np.zeros(n_pairs)
    pair_idx = 0
    for i in range(n):
        xi = 2*i
        yi = 2*i + 1
        ri = 2*n + i
        for j in range(i + 1, n):
            xj = 2*j
            yj = 2*j + 1
            rj = 2*n + j
            dx = v[xi] - v[xj]
            dy = v[yi] - v[yj]
            dr = v[ri] + v[rj]
            overlaps[pair_idx] = dx*dx + dy*dy - dr*dr
            pair_idx += 1
            
    return np.concatenate([bounds, overlaps])

def post_process(centers, radii):
    """Ensure all constraints are strictly satisfied."""
    n = len(radii)
    # 1. Enforce boundary constraints by shrinking radii
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        min_dist = min(x, 1.0-x, y, 1.0-y)
        if r > min_dist:
            radii[i] = min_dist
            
    # 2. Resolve overlaps by proportionally scaling down radii
    for i in range(n):
        for j in range(i+1, n):
            dist = math.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
            if dist < radii[i] + radii[j]:
                sum_r = radii[i] + radii[j]
                if sum_r > 0:
                    scale = dist / sum_r
                    radii[i] *= scale
                    radii[j] *= scale
    return centers, radii

def run_packing():
    n = 26
    # 1. Generate hexagonal initial layout
    centers = np.zeros((n, 2))
    idx = 0
    row = 0
    # Vertical spacing for hexagonal packing (approx sqrt(3)/2 * diameter)
    dy = 0.16
    y_start = 0.12
    
    while idx < n:
        if row % 2 == 0:
            xs = np.array([0.15, 0.35, 0.55, 0.75, 0.95])
        else:
            xs = np.array([0.25, 0.45, 0.65, 0.85])
        for x in xs:
            if idx >= n:
                break
            centers[idx] = [x, y_start + row * dy]
            idx += 1
        row += 1
        
    # Clamp initial positions to be safely inside
    centers = np.clip(centers, 0.1, 0.9)
    
    # 2. Add fixed perturbation to break symmetry and aid optimization
    for i in range(n):
        centers[i][0] += 0.008 * (i % 3)
        centers[i][1] += 0.008 * ((i // 3) % 3)
        
    # Initial radii
    radii = np.full(n, 0.085)
    x0 = np.concatenate([centers.ravel(), radii])
    
    # 3. Setup optimization
    bounds_opt = [(0.0, 1.0)] * (2*n) + [(0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Run SLSQP optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt, constraints=cons, 
                   options={'maxiter': 5000, 'ftol': 1e-9})
                   
    opt_centers = res.x[:2*n].reshape(n, 2)
    opt_radii = res.x[2*n:]
    
    # 4. Post-process to guarantee validity
    opt_centers, opt_radii = post_process(opt_centers, opt_radii)
    
    total_r = np.sum(opt_radii)
    return opt_centers, opt_radii, total_r