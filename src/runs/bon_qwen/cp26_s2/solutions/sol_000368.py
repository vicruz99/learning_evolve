# sol_000368 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b75b923f) state=52e42b92 sum of radii=2.436634 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(params, n):
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    penalty = 0.0
    
    # Vectorized overlap penalty
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    sum_radii = radii[:, None] + radii[None, :]
    overlaps = sum_radii - dists
    np.fill_diagonal(overlaps, -1e9)
    overlaps = np.triu(overlaps)
    penalty += np.sum(np.maximum(overlaps, 0.0)**2)
    
    # Vectorized boundary penalty
    x, y = centers[:, 0], centers[:, 1]
    penalty += np.sum(np.minimum(x - radii, 0.0)**2)
    penalty += np.sum(np.minimum(1.0 - x - radii, 0.0)**2)
    penalty += np.sum(np.minimum(y - radii, 0.0)**2)
    penalty += np.sum(np.minimum(1.0 - y - radii, 0.0)**2)
        
    return -np.sum(radii) + 5000.0 * penalty

def run_packing():
    n = 26
    # Initial hexagonal arrangement
    coords = []
    y_vals = np.linspace(0.15, 0.85, 5)
    counts = [5, 6, 5, 6, 4]
    
    for i, y in enumerate(y_vals):
        if counts[i] > 0:
            x_vals = np.linspace(0.1, 0.9, counts[i])
            if i % 2 == 1:
                dx = x_vals[1] - x_vals[0]
                x_vals = x_vals + dx / 2.0
            for x in x_vals:
                coords.append([x, y])
                
    if len(coords) != n:
        coords = np.random.uniform(0.2, 0.8, (n, 2))
    else:
        coords = np.array(coords)
        
    # Add slight random perturbation to break symmetry
    coords += np.random.uniform(-0.005, 0.005, coords.shape)
    
    r_init = np.full(n, 0.08)
    x0 = np.concatenate([coords.flatten(), r_init])
    
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)])
    bounds.extend([(1e-6, 0.5) for _ in range(n)])
    
    res = minimize(compute_objective, x0, args=(n,), method='L-BFGS-B', 
                   bounds=bounds, options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-12})
                   
    centers = res.x[:2*n].reshape(n, 2)
    radii = res.x[2*n:]
    
    # Post-processing to ensure strict feasibility
    for _ in range(300):
        changed = False
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            new_r = min(r, x, 1.0 - x, y, 1.0 - y)
            if new_r < r - 1e-9:
                radii[i] = new_r
                changed = True
                
        # Pairwise constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if sum_r > dist + 1e-9:
                    scale = dist / sum_r
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
            
    # Uniform scale up to maximum feasible
    max_scale = 1.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r > 1e-12:
            max_scale = min(max_scale, x/r, (1.0 - x)/r, y/r, (1.0 - y)/r)
            
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = radii[i] + radii[j]
            if sum_r > 1e-12:
                max_scale = min(max_scale, dist / sum_r)
                
    radii *= max_scale
    radii *= 0.999999 # Safety margin for numerical validation
    
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
