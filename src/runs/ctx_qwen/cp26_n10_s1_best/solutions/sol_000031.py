# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state 9b0797fd) state=b051e300 sum of radii=2.626968 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_func(vars):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars[52:])

def compute_constraints(vars):
    """Compute all boundary and non-overlap constraints as a single vector."""
    n = 26
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    parts = []
    
    # Boundary constraints: g(vars) >= 0
    # x >= r  => x - r >= 0
    parts.append(centers[:, 0] - radii)
    # x <= 1 - r => 1 - x - r >= 0
    parts.append(1.0 - centers[:, 0] - radii)
    # y >= r => y - r >= 0
    parts.append(centers[:, 1] - radii)
    # y <= 1 - r => 1 - y - r >= 0
    parts.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap: dist(i, j) >= r_i + r_j
    # Vectorized distance computation
    c1 = centers[:, np.newaxis, :]
    c2 = centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum((c1 - c2)**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Lower triangle indices to avoid duplicates and self-comparison
    tri = np.tril_indices(n, -1)
    parts.append(dists[tri] - r_sum[tri])
    
    return np.concatenate(parts)

def run_packing():
    n = 26
    num_vars = 3 * n
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    constraints = {'type': 'ineq', 'fun': compute_constraints}
    
    best_sum = -1.0
    best_vars = None
    
    # Helper to generate diverse initial guesses
    def make_init(seed):
        np.random.seed(seed)
        r_init = 0.09
        centers = []
        count = 0
        y = r_init
        # Hexagonal grid generation
        while count < n:
            x = r_init
            while count < n:
                centers.append([x, y])
                count += 1
                x += 2 * r_init
                if x + r_init > 1.0:
                    break
            y += np.sqrt(3) * r_init
            if y + r_init > 1.0:
                break
        
        # Fallback if grid didn't fill (shouldn't happen with r=0.09)
        while count < n:
            r = np.random.uniform(0.05, 0.15)
            centers.append([np.random.uniform(r, 1-r), np.random.uniform(r, 1-r)])
            count += 1
            
        centers = np.array(centers[:n])
        radii = np.full(n, r_init)
        
        # Add controlled noise to escape symmetries
        centers += np.random.normal(0, 0.025, centers.shape)
        radii += np.random.normal(0, 0.015, radii.shape)
        radii = np.clip(radii, 0.02, 0.4)
        
        # Project to feasible bounds roughly
        centers[:, 0] = np.clip(centers[:, 0], radii, 1 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1 - radii)
        
        return np.concatenate([centers.flatten(), radii])

    # Phase 1: Multi-start optimization
    for seed in range(40):
        v0 = make_init(seed)
        try:
            res = minimize(obj_func, v0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_vars = res.x.copy()
        except Exception:
            pass
            
    # Fallback if optimization completely failed
    if best_vars is None:
        best_vars = make_init(0)
        
    # Phase 2: Local search refinement around the best found solution
    # Perturb and re-optimize to escape shallow local minima
    for _ in range(10):
        noisy_vars = best_vars + np.random.normal(0, 1e-4, size=num_vars)
        try:
            res = minimize(obj_func, noisy_vars, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_vars = res.x.copy()
        except Exception:
            pass
            
    centers = best_vars[:2*n].reshape(n, 2)
    radii = best_vars[2*n:]
    
    # Phase 3: Strict validity check and numerical repair
    valid = True
    for i in range(n):
        if radii[i] < 0 or centers[i,0] < radii[i] - 1e-12 or centers[i,0] + radii[i] > 1 + 1e-12 or \
           centers[i,1] < radii[i] - 1e-12 or centers[i,1] + radii[i] > 1 + 1e-12:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Aggressively shrink radii until strictly valid
        for _ in range(50):
            radii *= 0.99
            valid = True
            for i in range(n):
                if centers[i,0] < radii[i] - 1e-12 or centers[i,0] + radii[i] > 1 + 1e-12 or \
                   centers[i,1] < radii[i] - 1e-12 or centers[i,1] + radii[i] > 1 + 1e-12:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                        if d < radii[i] + radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                break
                
    return centers, radii, float(np.sum(radii))
