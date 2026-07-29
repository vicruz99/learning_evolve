# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state 1cb5ec92) state=fed03795 sum of radii=2.154837 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(v, n):
    """Minimize negative sum of radii to maximize total radius."""
    return -np.sum(v[2*n:])

def constraint_func(v, n):
    """Compute all boundary and non-overlap constraints (must be >= 0)."""
    centers = v[:2*n].reshape(n, 2)
    radii = v[2*n:]
    
    c = []
    # Boundary constraints
    c.append(centers[:, 0] - radii)
    c.append(1.0 - centers[:, 0] - radii)
    c.append(centers[:, 1] - radii)
    c.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap constraints (vectorized)
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sums = radii[:, None] + radii[None, :]
    
    mask = np.triu_indices(n, k=1)
    c.append(dists[mask] - r_sums[mask])
    
    # Non-negative radii
    c.append(radii)
    
    return np.concatenate(c)

def generate_init(n, seed):
    """Generate initial centers using a perturbed hexagonal lattice."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 2))
    idx = 0
    y = 0.1
    row = 0
    while idx < n:
        x_start = 0.1 if row % 2 == 0 else 0.1732
        x = x_start
        while x <= 0.9 and idx < n:
            centers[idx, 0] = x + rng.uniform(-0.02, 0.02)
            centers[idx, 1] = y + rng.uniform(-0.02, 0.02)
            idx += 1
            x += 0.1732
        y += 0.15
        row += 1
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.full(n, 0.08)
    return centers, radii

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (n,)}
    
    best_sum = -1.0
    best_v = None
    
    # Multi-start optimization
    for trial in range(50):
        if trial < 15:
            centers, radii = generate_init(n, seed=trial)
        else:
            if best_v is not None:
                centers = best_v[:2*n].reshape(n, 2)
                radii = best_v[2*n:]
                centers += np.random.normal(0, 0.004, centers.shape)
                centers = np.clip(centers, 0.05, 0.95)
                radii = np.clip(radii, 0.01, 0.5)
            else:
                centers, radii = generate_init(n, seed=trial)
                
        x0 = np.concatenate([centers.flatten(), radii])
        
        try:
            res = minimize(objective_func, x0, args=(n,), method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
            
            cur_sum = -res.fun
            # Verify constraint satisfaction with tolerance
            cons_vals = constraint_func(res.x, n)
            if np.min(cons_vals) >= -1e-6 and cur_sum > best_sum:
                best_sum = cur_sum
                best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        centers, radii = generate_init(n, seed=0)
        best_v = np.concatenate([centers.flatten(), radii])
        best_sum = np.sum(radii)
        
    centers_opt = best_v[:2*n].reshape(n, 2)
    radii_opt = best_v[2*n:]
    
    # Post-processing to strictly enforce constraints and resolve numerical overlaps
    for _ in range(10):
        overlap = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers_opt[i] - centers_opt[j])
                if d < radii_opt[i] + radii_opt[j] - 1e-9:
                    shrink = (radii_opt[i] + radii_opt[j] - d) * 0.51
                    radii_opt[i] -= shrink
                    radii_opt[j] -= shrink
                    overlap = True
            if centers_opt[i, 0] - radii_opt[i] < 1e-9:
                radii_opt[i] = max(0, centers_opt[i, 0] - 1e-10)
            if 1.0 - centers_opt[i, 0] - radii_opt[i] < 1e-9:
                radii_opt[i] = max(0, 1.0 - centers_opt[i, 0] - 1e-10)
            if centers_opt[i, 1] - radii_opt[i] < 1e-9:
                radii_opt[i] = max(0, centers_opt[i, 1] - 1e-10)
            if 1.0 - centers_opt[i, 1] - radii_opt[i] < 1e-9:
                radii_opt[i] = max(0, 1.0 - centers_opt[i, 1] - 1e-10)
        if not overlap:
            break
            
    radii_opt = np.maximum(radii_opt, 0.0)
    final_sum = float(np.sum(radii_opt))
    
    return centers_opt, radii_opt, final_sum
