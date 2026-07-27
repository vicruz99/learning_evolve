# sol_000153 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 39a6f529) state=4c2b1ad2 sum of radii=1.919920 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(centers, radii, n):
    """Compute penalty for boundary violations and overlaps."""
    pen = 0.0
    # Boundary penalties
    for i in range(n):
        r = radii[i]
        x, y = centers[i]
        pen += max(0.0, r - x) ** 2
        pen += max(0.0, r - (1.0 - x)) ** 2
        pen += max(0.0, r - y) ** 2
        pen += max(0.0, r - (1.0 - y)) ** 2
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0.0:
                pen += overlap ** 2
    return pen

def objective(vars, n):
    """Objective function: minimize -sum(radii) + penalty."""
    centers = vars[:2 * n].reshape(n, 2)
    radii = vars[2 * n:]
    pen = compute_penalty(centers, radii, n)
    return -np.sum(radii) + 100000.0 * pen

def check_validity(centers, radii, n, tol=1e-10):
    """Check if packing is valid within tolerance."""
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def run_packing():
    """Pack 26 circles in a unit square to maximize sum of radii."""
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # Try multiple initial configurations
    for seed in range(5):
        rng = np.random.default_rng(seed)
        centers = np.zeros((n, 2))
        idx = 0
        
        # Hexagonal-ish initial layout
        for row in range(6):
            y = 0.1 + row * 0.15
            x_start = 0.1 + (0.0 if row % 2 == 0 else 0.075)
            for col in range(5):
                if idx < n:
                    centers[idx] = [x_start + col * 0.15, y]
                    idx += 1
                    
        # Perturb to break symmetry and avoid flat minima
        centers += rng.uniform(-0.02, 0.02, size=centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        radii = np.full(n, 0.05)
        
        x0 = np.concatenate([centers.flatten(), radii])
        
        res = minimize(
            objective, 
            x0, 
            args=(n,), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        c_ = res.x[:2 * n].reshape(n, 2)
        r_ = res.x[2 * n:]
        current_sum = np.sum(r_)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = c_.copy()
            best_radii = r_.copy()
            
    # Post-processing: ensure strict validity for the validator
    while not check_validity(best_centers, best_radii, n, tol=1e-9):
        best_radii *= 0.999
        if np.sum(best_radii) < 0.1:
            break
            
    final_sum = np.sum(best_radii)
    return best_centers, best_radii, final_sum
