# sol_000042 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000031 (state b051e300) state=826d2244 sum of radii=2.625918 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective to minimize: negative sum of radii."""
    return -np.sum(vars[52:])

def compute_constraints(vars):
    """Compute all boundary and non-overlap constraints as a single vector >= 0."""
    n = len(vars) // 3
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    parts = [
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ]
    
    c1 = centers[:, np.newaxis, :]
    c2 = centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum((c1 - c2)**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    tri = np.tril_indices(n, -1)
    parts.append(dists[tri] - r_sum[tri])
    
    return np.concatenate(parts)

def run_packing():
    n = 26
    num_vars = 3 * n
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_sum = -1.0
    best_vars = None
    
    # Helper to create initial configurations
    inits = []
    
    # 1. Hexagonal lattice patterns with varying row distributions
    patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 6, 6, 5, 4],
        [4, 6, 6, 6, 4],
        [6, 6, 5, 6, 3],
        [5, 5, 5, 5, 5, 1]
    ]
    
    for pat in patterns:
        r0 = 0.092
        centers = []
        y = r0
        row = 0
        for cnt in pat:
            x = r0 + (r0 if row % 2 == 1 else 0)
            for _ in range(cnt):
                centers.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
            
        centers = np.array(centers[:n])
        radii = np.full(n, r0)
        
        # Controlled perturbation to break symmetry
        centers += np.random.uniform(-0.006, 0.006, centers.shape)
        radii += np.random.uniform(-0.006, 0.006, radii.shape)
        radii = np.clip(radii, 0.04, 0.25)
        
        # Project to feasible bounds
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        inits.append(np.concatenate([centers.flatten(), radii]))
        
    # 2. Random layouts with overlap resolution
    for seed in range(25):
        np.random.seed(seed + 100)
        radii = np.random.uniform(0.055, 0.115, n)
        centers = np.zeros((n, 2))
        for i in range(n):
            centers[i] = np.random.uniform(radii[i], 1.0 - radii[i], 2)
            
        # Quick force relaxation to resolve overlaps
        for _ in range(150):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d = np.hypot(dx, dy)
                    min_d = radii[i] + radii[j]
                    if d < min_d and d > 1e-7:
                        shift = (min_d - d) * 0.5 / d
                        centers[i, 0] -= shift * dx
                        centers[i, 1] -= shift * dy
                        centers[j, 0] += shift * dx
                        centers[j, 1] += shift * dy
                        moved = True
            if not moved:
                break
                
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        inits.append(np.concatenate([centers.flatten(), radii]))
        
    # Phase 1: Multi-start optimization
    for v0 in inits:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            s = -res.fun
            vals = compute_constraints(res.x)
            if np.all(vals >= -1e-7) and s > best_sum:
                best_sum = s
                best_vars = res.x.copy()
        except Exception:
            pass
            
    # Fallback if all optimizations fail
    if best_vars is None:
        best_vars = inits[0]
        best_sum = -objective(best_vars)
        
    # Phase 2: Local refinement to escape shallow local minima
    for _ in range(12):
        noisy_vars = best_vars + np.random.normal(0, 1.5e-4, size=num_vars)
        noisy_vars[0::3] = np.clip(noisy_vars[0::3], 0.0, 1.0)
        noisy_vars[1::3] = np.clip(noisy_vars[1::3], 0.0, 1.0)
        noisy_vars[2::3] = np.clip(noisy_vars[2::3], 0.0, 0.5)
        
        try:
            res = minimize(objective, noisy_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            s = -res.fun
            vals = compute_constraints(res.x)
            if np.all(vals >= -1e-7) and s > best_sum:
                best_sum = s
                best_vars = res.x.copy()
        except Exception:
            pass
            
    # Extract results
    centers = best_vars[:2*n].reshape(n, 2)
    radii = best_vars[2*n:]
    
    # Phase 3: Strict validity check and numerical repair
    valid = True
    for i in range(n):
        if (radii[i] < 0 or 
            centers[i, 0] < radii[i] - 1e-12 or centers[i, 0] + radii[i] > 1.0 + 1e-12 or 
            centers[i, 1] < radii[i] - 1e-12 or centers[i, 1] + radii[i] > 1.0 + 1e-12):
            valid = False
            break
            
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Minimal shrinkage to guarantee strict compliance
        for _ in range(100):
            radii *= 0.999
            valid = True
            for i in range(n):
                if (centers[i, 0] < radii[i] - 1e-12 or centers[i, 0] + radii[i] > 1.0 + 1e-12 or 
                    centers[i, 1] < radii[i] - 1e-12 or centers[i, 1] + radii[i] > 1.0 + 1e-12):
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                        if d < radii[i] + radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                break
                
    return centers, radii, float(np.sum(radii))
