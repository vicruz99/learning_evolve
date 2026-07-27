import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26

    # 1. Initialization: Perturbed Hexagonal Grid
    centers, radii = generate_hexagonal_guess(n)
    
    # 2. Optimization
    centers, radii = optimize_packing(centers, radii, n)
    
    # 3. Validation & Safety Cleanup
    centers, radii = ensure_validity(centers, radii)

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

def generate_hexagonal_guess(n):
    """Generates initial centers and radii using a perturbed hexagonal grid."""
    # Hex grid parameters
    r_init = 0.09
    centers = []
    y = r_init
    while len(centers) < n:
        x = r_init
        offset = r_init if len(centers) % 2 == 1 else r_init * 0.5
        x = offset
        while x <= 1 - r_init:
            if len(centers) < n:
                # Add slight random perturbation to avoid local minima
                cx = x + np.random.uniform(-0.01, 0.01)
                cy = y + np.random.uniform(-0.01, 0.01)
                centers.append([cx, cy])
            x += np.sqrt(3) * r_init
        y += 1.5 * r_init
    
    centers = np.array(centers[:n])
    radii = np.full(n, r_init)
    return centers, radii

def objective_to_minimize(vars_flat, n):
    """Objective function: Negate sum of radii to perform maximization."""
    radii = vars_flat[2::3]
    return -np.sum(radii)

def get_constraints(n):
    """Generates boundary and overlap constraints for SLSQP."""
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        idx_x, idx_y, idx_r = 3 * i, 3 * i + 1, 3 * i + 2
        
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_x] - v[idx_r]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[idx_x] - v[idx_r]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_y] - v[idx_r]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[idx_y] - v[idx_r]})

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = 3 * i, 3 * i + 1, 3 * i + 2
            idx_xj, idx_yj, idx_rj = 3 * j, 3 * j + 1, 3 * j + 2
            
            def dist_constraint(v, i=i, j=j):
                dist_sq = (v[idx_xi] - v[idx_xj])**2 + (v[idx_yi] - v[idx_yj])**2
                sum_r = v[idx_ri] + v[idx_rj]
                return dist_sq - sum_r**2 # Must be >= 0
            
            constraints.append({'type': 'ineq', 'fun': dist_constraint})
            
    return constraints

def optimize_packing(centers, radii, n):
    """Performs the optimization to maximize the sum of radii."""
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.array([val for pair in zip(centers[:, 0], centers[:, 1], radii) for val in pair])
    
    bounds = [(0, 1) for _ in range(2 * n)] + [(0, 0.5) for _ in range(n)]
    
    # Run optimization with SLSQP
    result = minimize(objective_to_minimize, x0, args=(n,), method='SLSQP',
                      bounds=bounds, constraints=get_constraints(n),
                      options={'maxiter': 500, 'ftol': 1e-8})
    
    opt_vars = result.x
    centers_opt = np.array([opt_vars[i::3] for i in range(2)]).T
    radii_opt = opt_vars[2::3]
    
    return centers_opt, radii_opt

def ensure_validity(centers, radii):
    """Post-processing step to ensure strict non-overlap and boundary containment."""
    n = len(radii)
    
    # 1. Enforce boundaries (clamp and shrink if necessary)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        # Clamp centers to valid range
        x = max(r, min(1 - r, x))
        y = max(r, min(1 - r, y))
        centers[i] = [x, y]
        
        # Shrink if center is out of bounds (safety)
        if x < r: radii[i] = x
        if x > 1 - r: radii[i] = 1 - x
        if y < r: radii[i] = y
        if y > 1 - r: radii[i] = 1 - y

    # 2. Enforce non-overlap (shrink larger circle if overlap detected)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            sum_r = radii[i] + radii[j]
            
            if dist < sum_r:
                # Overlap detected. Shrink radii to just touch.
                excess = sum_r - dist
                # Distribute shrinkage proportionally or just halve it
                # Here we halve the excess for stability
                radii[i] -= excess / 2
                radii[j] -= excess / 2
                
    # Ensure non-negative radii after adjustments
    radii = np.maximum(radii, 1e-9)
    
    return centers, radii