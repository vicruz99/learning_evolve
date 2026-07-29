# sol_000034 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=e85e6340 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
PAIR_I = np.array([i for i in range(N_CIRCLES) for j in range(i+1, N_CIRCLES)])
PAIR_J = np.array([j for i in range(N_CIRCLES) for j in range(i+1, N_CIRCLES)])

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N_CIRCLES:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap."""
    centers = v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = v[2*N_CIRCLES:]
    
    # Boundary constraints: x >= r, x + r <= 1, y >= r, y + r <= 1
    c1 = centers[:, 0] - radii
    c2 = 1.0 - centers[:, 0] - radii
    c3 = centers[:, 1] - radii
    c4 = 1.0 - centers[:, 1] - radii
    
    # Pairwise non-overlap: dist >= r_i + r_j
    diff = centers[PAIR_I] - centers[PAIR_J]
    dist = np.sqrt(np.sum(diff**2, axis=1) + 1e-16)  # 1e-16 prevents NaN gradients
    c5 = dist - radii[PAIR_I] - radii[PAIR_J]
    
    return np.concatenate([c1, c2, c3, c4, c5])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    
    # Multi-start optimization with diverse initializations
    for trial in range(30):
        if trial < 20:
            # Hexagonal lattice initialization with varied spacing
            r_init = 0.07 + np.random.uniform(0.0, 0.025)
            pts = []
            y = r_init
            row = 0
            while len(pts) < N_CIRCLES + 10:
                x = r_init + (row % 2) * r_init
                while x < 1 - r_init and len(pts) < N_CIRCLES + 10:
                    pts.append([x, y])
                    x += 2 * r_init
                y += r_init * np.sqrt(3)
                row += 1
            centers = np.array(pts[:N_CIRCLES])
        else:
            # Random initialization to break lattice symmetry
            centers = np.random.uniform(0.05, 0.95, size=(N_CIRCLES, 2))
            
        # Add jitter to avoid exact symmetries and guarantee feasible start
        centers += np.random.uniform(-0.03, 0.03, size=centers.shape)
        centers = np.clip(centers, 0.01, 0.99)
        radii = np.full(N_CIRCLES, 0.02)
        v0 = np.concatenate([centers.flatten(), radii])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-12})
            if res.success:
                cur_sum = -res.fun
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_centers = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2).copy()
                    best_radii = res.x[2*N_CIRCLES:].copy()
        except Exception:
            pass
            
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = np.random.rand(N_CIRCLES, 2)
        best_radii = np.full(N_CIRCLES, 0.01)
        
    # Strict validity enforcement to satisfy 1e-12 validator tolerance
    radii = best_radii.copy()
    centers = best_centers.copy()
    radii = np.maximum(radii, 0.0)
    
    for _ in range(5):
        for i in range(N_CIRCLES):
            # Max radius allowed by walls
            r_lim = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
            # Max radius allowed by other circles
            for j in range(N_CIRCLES):
                if i == j: continue
                d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                r_lim = min(r_lim, d - radii[j])
            radii[i] = min(radii[i], max(0.0, r_lim))
            
    return centers, radii, float(np.sum(radii))
