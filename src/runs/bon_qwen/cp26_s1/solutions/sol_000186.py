# sol_000186 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9afef83a) state=9fb020d6 sum of radii=0.130567 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(vars):
    """Maximize radius r (minimize -r)"""
    return -vars[-1]

def _constraints(vars):
    """
    Inequality constraints: g(vars) >= 0
    - Boundary constraints for each circle
    - Non-overlap constraints for each pair
    """
    n = 26
    cx = vars[:2*n].reshape(n, 2)
    r = vars[2*n]
    
    res = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, same for y
    res.append(cx[:, 0] - r)
    res.append(1.0 - cx[:, 0] - r)
    res.append(cx[:, 1] - r)
    res.append(1.0 - cx[:, 1] - r)
    
    # Non-overlap constraints: distance >= 2r
    diff = cx[:, np.newaxis, :] - cx[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    res.append(dists[np.triu_indices(n, 1)] - 2.0 * r)
    
    return np.concatenate(res)

def _generate_initial_guess(n):
    """Generate a hexagonal-like initial configuration"""
    pts = []
    for i in range(12):
        for j in range(12):
            x = j * 0.14 + (i % 2) * 0.07
            y = i * 0.125
            pts.append([x, y])
        if len(pts) >= n:
            break
            
    pts = np.array(pts[:n])
    # Normalize to fit comfortably inside [0,1]^2
    pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0)) * 0.7 + 0.15
    return pts

def run_packing():
    n = 26
    best_r = 0.0
    best_centers = None
    
    # Try multiple random seeds to escape local optima
    for seed in [42, 123, 456, 789, 1024]:
        np.random.seed(seed)
        pts = _generate_initial_guess(n)
        
        # Add small perturbation
        pts += np.random.randn(n, 2) * 0.002
        pts = np.clip(pts, 0.05, 0.95)
        
        x0 = np.concatenate([pts.flatten(), [0.045]])
        bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)]
        cons = {'type': 'ineq', 'fun': _constraints}
        
        try:
            res = minimize(_objective, x0, bounds=bounds, constraints=cons,
                           method='SLSQP', options={'maxiter': 400, 'ftol': 1e-10})
            if res.x[-1] > best_r:
                best_r = res.x[-1]
                best_centers = res.x[:2*n].reshape(n, 2)
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = _generate_initial_guess(n)
        best_r = 0.045
        
    # Numerical safety margin to strictly satisfy validation constraints
    dists = np.sqrt(np.sum((best_centers[:, None, :] - best_centers[None, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_coord = np.min(best_centers)
    max_coord = np.max(best_centers)
    min_boundary_dist = min(min_coord, 1.0 - max_coord)
    
    # Clamp radius to ensure strict validity
    safety = 1e-7
    best_r = min(best_r, min_pair_dist / 2.0 - safety, min_boundary_dist - safety)
    
    radii = np.full(n, best_r)
    return best_centers, radii, best_r * n
