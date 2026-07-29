# sol_000026 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e154041e) state=8e0dd625 sum of radii=2.592939 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def objective(v):
    # Maximize sum of radii => Minimize -sum of radii
    return -np.sum(v[:26])

def make_lower_bound(r_idx, pos_idx):
    def constraint(v):
        return v[pos_idx] - v[r_idx]
    return constraint

def make_upper_bound(r_idx, pos_idx):
    def constraint(v):
        return 1.0 - v[pos_idx] - v[r_idx]
    return constraint

def make_radius_nonneg(r_idx):
    def constraint(v):
        return v[r_idx]
    return constraint

def make_overlap(i_x, i_y, i_r, j_x, j_y, j_r):
    def constraint(v):
        return (v[i_x] - v[j_x])**2 + (v[i_y] - v[j_y])**2 - (v[i_r] + v[j_r])**2
    return constraint

def run_packing():
    n = 26
    
    # Initial guess: 5x5 grid + one circle in a gap
    centers = []
    for r in range(5):
        for c in range(5):
            centers.append([0.1 + c * 0.2, 0.1 + r * 0.2])
    centers.append([0.2, 0.2]) # 26th circle
    centers = np.array(centers)
    
    # Start with a valid radius that is small enough to be strictly valid
    radii_init = np.full(n, 0.05)
    
    # Vector structure: [r_0...r_25, x_0, y_0... x_25, y_25]
    x0 = np.concatenate([radii_init, centers.flatten()])
    
    constraints = []
    
    for i in range(n):
        r_idx = i
        x_idx = n + 2*i
        y_idx = n + 2*i + 1
        
        # Boundary constraints
        constraints.append({'type': 'ineq', 'fun': make_lower_bound(r_idx, x_idx)})
        constraints.append({'type': 'ineq', 'fun': make_upper_bound(r_idx, x_idx)})
        constraints.append({'type': 'ineq', 'fun': make_lower_bound(r_idx, y_idx)})
        constraints.append({'type': 'ineq', 'fun': make_upper_bound(r_idx, y_idx)})
        constraints.append({'type': 'ineq', 'fun': make_radius_nonneg(r_idx)})
        
        # Overlap constraints
        for j in range(i + 1, n):
            r_j_idx = j
            x_j_idx = n + 2*j
            y_j_idx = n + 2*j + 1
            constraints.append({
                'type': 'ineq', 
                'fun': make_overlap(x_idx, y_idx, r_idx, x_j_idx, y_j_idx, r_j_idx)
            })
            
    bounds = [(0, 0.5) for _ in range(n)] + [(0, 1) for _ in range(2*n)]
    
    # Run optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 5000, 'ftol': 1e-12})
    
    opt_radii = res.x[:n]
    opt_centers = res.x[n:].reshape((n, 2))
    
    # Post-optimization validation and correction for numerical precision
    if not validate_packing(opt_centers, opt_radii):
        scale = 0.9999
        while not validate_packing(opt_centers, opt_radii * scale):
            scale *= 0.9999
        opt_radii *= scale
        
    return opt_centers, opt_radii, np.sum(opt_radii)
