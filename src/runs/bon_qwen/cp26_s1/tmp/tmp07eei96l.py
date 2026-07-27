import numpy as np
import scipy.optimize as opt

def objective(vars):
    """
    Objective function to minimize: negative sum of radii.
    vars: flattened array [x1, y1, x2, y2, ..., r1, r2, ...]
    """
    n = len(vars) // 3
    radii = vars[2*n:]
    return -np.sum(radii)

def compute_constraints(vars):
    """
    Computes all inequality constraints.
    Returns a concatenated array of constraint values.
    Constraints are satisfied if all values >= 0.
    """
    n = len(vars) // 3
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    cons = []
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    # x - r >= 0
    cons.append(centers[:, 0] - radii)
    # 1 - x - r >= 0
    cons.append(1.0 - centers[:, 0] - radii)
    # y - r >= 0
    cons.append(centers[:, 1] - radii)
    # 1 - y - r >= 0
    cons.append(1.0 - centers[:, 1] - radii)
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    # Vectorized computation for all pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap = dist_sq - r_sum**2
    
    # We only need constraints for i < j (upper triangle)
    idx = np.triu_indices(n, k=1)
    cons.append(overlap[idx])
    
    return np.concatenate(cons)

def run_packing():
    n = 26
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    best_x = None
    max_sum = -1.0
    
    # Define initial configurations to try
    
    # 1. 5x5 Grid configuration
    gx = np.linspace(0.1, 0.9, 5)
    gy = np.linspace(0.1, 0.9, 5)
    cx, cy = np.meshgrid(gx, gy)
    grid_centers = np.vstack([cx.ravel(), cy.ravel()]).T
    
    # Try placing the 26th circle in various gaps
    # Gaps are at coordinates like (0.2, 0.2), (0.2, 0.4), etc.
    gaps = [
        [0.2, 0.2], [0.2, 0.4], [0.4, 0.2], 
        [0.8, 0.8], [0.6, 0.4], [0.4, 0.6],
        [0.5, 0.2], [0.2, 0.5]
    ]
    
    for gap in gaps:
        centers = np.vstack([grid_centers, gap])
        # Start with radius 0.09 for grid circles (valid since spacing 0.2 > 0.18)
        # and a smaller radius for the gap circle to ensure feasibility
        radii = np.full(n, 0.09)
        radii[-1] = 0.03 
        x0 = np.concatenate([centers.ravel(), radii])
        
        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                               constraints={'type': 'ineq', 'fun': compute_constraints},
                               options={'maxiter': 2000, 'ftol': 1e-12})
            
            # Check if result is feasible (constraints satisfied)
            # Allow small numerical tolerance
            cons_val = compute_constraints(res.x)
            if np.min(cons_val) >= -1e-6:
                current_sum = -res.fun
                if current_sum > max_sum:
                    max_sum = current_sum
                    best_x = res.x.copy()
        except Exception:
            pass

    # 2. Random configurations to escape local minima
    # Use small initial radii to ensure feasibility
    for _ in range(5):
        centers = np.random.rand(n, 2)
        radii = np.full(n, 0.02)
        x0 = np.concatenate([centers.ravel(), radii])
        
        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                               constraints={'type': 'ineq', 'fun': compute_constraints},
                               options={'maxiter': 2000, 'ftol': 1e-12})
            
            cons_val = compute_constraints(res.x)
            if np.min(cons_val) >= -1e-6:
                current_sum = -res.fun
                if current_sum > max_sum:
                    max_sum = current_sum
                    best_x = res.x.copy()
        except Exception:
            pass
            
    if best_x is not None:
        centers = best_x[:2*n].reshape(n, 2)
        radii = best_x[2*n:]
        # Ensure non-negative radii and valid centers
        radii = np.maximum(radii, 0)
        centers = np.clip(centers, 0, 1)
        return centers, radii, np.sum(radii)
    else:
        # Fallback to a valid configuration if optimization fails
        centers = np.vstack([grid_centers, [0.2, 0.2]])
        radii = np.full(26, 0.05)
        return centers, radii, np.sum(radii)