# sol_000151 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3a06727e) state=4d1a7195 sum of radii=2.160534 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
PENALTY_WEIGHT = 8000.0

def compute_objective(vars, n, P):
    """Objective function: maximize sum of radii with quadratic penalties for violations."""
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Boundary penalties: circles must be inside [0,1]x[0,1]
    dL = c[:, 0]
    dR = 1.0 - c[:, 0]
    dB = c[:, 1]
    dT = 1.0 - c[:, 1]
    
    pen = np.sum(np.clip(r - dL, 0, None)**2 + 
                 np.clip(r - dR, 0, None)**2 + 
                 np.clip(r - dB, 0, None)**2 + 
                 np.clip(r - dT, 0, None)**2)
                 
    # Overlap penalties: distance between centers must be >= sum of radii
    dists = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    R = r[:, None] + r[None, :]
    overlaps = np.triu(np.clip(R - dists, 0, None), k=1)
    pen += np.sum(overlaps**2)
    
    return -np.sum(r) + P * pen

def post_process(c, r, n):
    """Strictly enforce constraints by clamping and resolving overlaps."""
    # Clamp to boundaries
    for i in range(n):
        max_r = min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1])
        if r[i] > max_r:
            r[i] = max_r
            
    # Resolve overlaps iteratively
    for _ in range(30):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < r[i] + r[j] - 1e-9:
                    factor = d / (r[i] + r[j])
                    r[i] *= factor
                    r[j] *= factor
                    changed = True
        if not changed:
            break
    return c, r

def run_packing():
    n = N_CIRCLES
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    
    for trial in range(5):
        # Initialize with a hexagonal lattice perturbed by small noise
        centers = np.zeros((n, 2))
        idx = 0
        y = 0.12
        shift = 0.0
        while idx < n and y < 0.9:
            x = 0.12
            while x < 0.9 and idx < n:
                centers[idx] = [x + shift + np.random.normal(0, 0.005), 
                                y + np.random.normal(0, 0.005)]
                idx += 1
                x += 0.2
            y += 0.17320508
            shift = 0.1 if shift == 0.0 else 0.0
            
        radii = np.full(n, 0.05 + np.random.uniform(0, 0.01, n))
        x0 = np.concatenate([centers.flatten(), radii])
        
        # Ensure initial guess is within bounds
        x0[:2*n] = np.clip(x0[:2*n], 0.01, 0.99)
        x0[2*n:] = np.clip(x0[2*n:], 0.01, 0.4)
        
        bounds = [(0.001, 0.999)] * (2*n) + [(0.001, 0.5)] * n
        
        res = minimize(compute_objective, x0, args=(n, PENALTY_WEIGHT), 
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1200, 'ftol': 1e-11, 'gtol': 1e-9})
        
        c = res.x[:2*n].reshape(n, 2)
        r = res.x[2*n:].copy()
        
        # Strictly enforce constraints
        c, r = post_process(c, r, n)
        
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_centers = c.copy()
            best_radii = r.copy()
            
    return best_centers, best_radii, float(best_sum)
