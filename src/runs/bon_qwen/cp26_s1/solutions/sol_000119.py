# sol_000119 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4ac25994) state=e795f8a7 sum of radii=2.626930 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def get_initial_params():
    """Create a feasible initial configuration using a hexagonal-like grid."""
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.ones(N_CIRCLES) * 0.08
    
    idx = 0
    # Arrange in 5 rows with hexagonal shifting
    for row in range(5):
        y = 0.12 + row * 0.18
        shift = 0.09 if row % 2 == 1 else 0.0
        cols = 5
        for col in range(cols):
            if idx >= N_CIRCLES: 
                break
            x = 0.12 + col * 0.18 + shift
            # Clamp to safe interior region initially
            centers[idx] = [np.clip(x, 0.1, 0.9), np.clip(y, 0.1, 0.9)]
            idx += 1
            
    # Place any remaining circles in central gaps if needed
    while idx < N_CIRCLES:
        centers[idx] = [0.5, 0.5]
        idx += 1
        
    return np.concatenate([centers.flatten(), radii])

def objective(params):
    """Maximize sum of radii => Minimize negative sum."""
    radii = params[2 * N_CIRCLES:]
    return -np.sum(radii)

def constraints(params):
    """Define inequality constraints: g(x) >= 0."""
    centers = params[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = params[2 * N_CIRCLES:]
    vals = []
    
    # Pairwise non-overlap: dist(i, j) >= r_i + r_j
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            vals.append(dist - radii[i] - radii[j])
            
    # Boundary containment: r <= x <= 1-r, r <= y <= 1-r
    for i in range(N_CIRCLES):
        x, y = centers[i]
        r = radii[i]
        vals.append(x - r)
        vals.append(1.0 - x - r)
        vals.append(y - r)
        vals.append(1.0 - y - r)
        
    return np.array(vals)

def run_packing():
    """Run optimization and return valid packing."""
    x0 = get_initial_params()
    
    # Bounds: centers in [0, 1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Optimize using SLSQP
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False}
    )
    
    centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = res.x[2 * N_CIRCLES:]
    
    # Post-process to strictly enforce constraints within tolerance
    for i in range(N_CIRCLES):
        r = radii[i]
        x, y = centers[i]
        # Ensure circle stays inside square
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r + 1e-9:
            r = max_r
        radii[i] = r
        centers[i] = [np.clip(x, r, 1.0 - r), np.clip(y, r, 1.0 - r)]
        
    # Ensure radii are strictly positive to avoid validation issues
    radii = np.maximum(radii, 1e-6)
    
    return centers, radii, float(np.sum(radii))
