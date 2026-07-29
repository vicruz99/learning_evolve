# sol_000193 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bafdbd7e) state=70856236 sum of radii=2.605603 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2 * n:])

def compute_constraints(x, n):
    """Compute all inequality constraints."""
    centers = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    
    cons = []
    
    # Boundary constraints: x_i >= r_i, 1-x_i >= r_i, y_i >= r_i, 1-y_i >= r_i
    for i in range(n):
        cx, cy = centers[i]
        r = radii[i]
        cons.append(cx - r)
        cons.append(1.0 - cx - r)
        cons.append(cy - r)
        cons.append(1.0 - cy - r)
        
    # Pairwise non-overlap constraints: ||c_i - c_j|| >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            cons.append(dist - (radii[i] + radii[j]))
            
    return np.array(cons)

def run_packing():
    n = 26
    
    # 1. Hexagonal lattice initialization for better starting geometry
    pts = []
    for i in range(6):
        for j in range(6):
            x = i * np.sqrt(3) + (j % 2) * (np.sqrt(3) / 2)
            y = j * 1.5
            pts.append([x, y])
            
    pts = np.array(pts)
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    span = max_pt - min_pt
    
    # Scale to fit in [0.1, 0.9] region
    scale = 0.8 / span
    centers_init = (pts - min_pt) * scale + 0.1
    centers_init = centers_init[:n]
    
    # Start with small feasible radii
    radii_init = np.full(n, 0.03)
    
    # Flatten to optimization variable vector
    x0 = np.concatenate([centers_init.flatten(), radii_init])
    
    # Bounds: centers in [0,1], radii in [1e-6, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Constraint dictionary for SLSQP
    constraint = {
        'type': 'ineq',
        'fun': lambda x: compute_constraints(x, n)
    }
    
    # Run optimization
    res = minimize(
        fun=lambda x: compute_objective(x, n),
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraint,
        options={'maxiter': 2000, 'ftol': 1e-10}
    )
    
    # Extract results
    centers = res.x[:2 * n].reshape(n, 2)
    radii = res.x[2 * n:]
    
    # Post-process to strictly satisfy constraints within tolerance
    radii = np.clip(radii, 1e-7, 0.5)
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    
    # Slight shrinkage to guarantee no numerical overlap after clipping
    min_gap = 1e-9
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            req = radii[i] + radii[j]
            if dist < req:
                # Reduce radii equally to satisfy constraint
                shrink = (req - dist) / 2.0 + min_gap
                radii[i] = max(radii[i] - shrink, 1e-7)
                radii[j] = max(radii[j] - shrink, 1e-7)
    
    # Re-clip centers after radius adjustment
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    
    total_sum = float(np.sum(radii))
    return centers, radii, total_sum
