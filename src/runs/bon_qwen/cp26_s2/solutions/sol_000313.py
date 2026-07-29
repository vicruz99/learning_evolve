# sol_000313 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1cbfbe8a) state=3dfd9761 sum of radii=2.099841 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def evaluate_constraints(vars):
    n = N_CIRCLES
    centers = vars[:2*n].reshape(n, 2)
    t = vars[-1]
    
    # Boundary constraints: x >= t, 1-x >= t, y >= t, 1-y >= t
    c = np.concatenate([
        centers[:, 0] - t,
        1.0 - centers[:, 0] - t,
        centers[:, 1] - t,
        1.0 - centers[:, 1] - t
    ])
    
    # Pairwise distance constraints: dist^2 >= t^2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c = np.concatenate([c, dist2[mask] - t**2])
    
    return c

def objective_function(vars):
    # Maximize t (last variable) -> minimize -t
    return -vars[-1]

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)]
    cons = {'type': 'ineq', 'fun': evaluate_constraints}
    
    best_t = 0
    best_centers = None
    
    # Multiple restarts with different perturbations to avoid local minima
    seeds = [42, 123, 456, 789, 999]
    for seed in seeds:
        # Hexagonal lattice base pattern
        pts = []
        for i in range(6):
            for j in range(5):
                x = j * 2 + (i % 2) * 1
                y = i * np.sqrt(3)
                pts.append([x, y])
        pts = np.array(pts[:n])
        
        # Normalize to unit square
        pts -= pts.min(axis=0)
        pts /= pts.max(axis=0)
        
        # Add controlled noise
        rng = np.random.default_rng(seed)
        pts += rng.uniform(-0.05, 0.05, size=pts.shape)
        pts = np.clip(pts, 0.01, 0.99)
        
        x0 = np.concatenate([pts.flatten(), [0.08]])
        
        try:
            res = minimize(
                objective_function, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
            )
            if res.success and -res.fun > best_t:
                best_t = -res.fun
                best_centers = res.x[:2*n].reshape(n, 2)
        except Exception:
            continue
            
    # Fallback initialization if optimization fails unexpectedly
    if best_centers is None:
        best_centers = np.zeros((n, 2))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < n:
                    best_centers[idx] = [0.1 + j*0.2, 0.1 + i*0.2]
                    idx += 1
        while idx < n:
            best_centers[idx] = [0.05 + idx*0.03, 0.05]
            idx += 1
            
    centers = best_centers
    
    # Compute strictly feasible radius from optimized centers
    r = np.min([
        np.min(centers[:, 0]),
        np.min(1.0 - centers[:, 0]),
        np.min(centers[:, 1]),
        np.min(1.0 - centers[:, 1])
    ])
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r = min(r, np.min(dist[mask]) / 2.0)
    
    radii = np.full(n, r)
    return centers, radii, np.sum(radii)
