# sol_000077 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000001 (state 1501c8b5) state=4326f526 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """
    Computes all inequality constraints:
    1. Boundary: center +/- radius within [0, 1]
    2. Pairwise: squared distance >= squared sum of radii
    Returns a flat array where each element >= 0 indicates a satisfied constraint.
    """
    cx = v[:N]
    cy = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints
    c = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    # Pairwise non-overlap constraints (vectorized)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, None] + r[None, :]
    
    # Upper triangle mask to avoid duplicate and self-constraints
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c = np.concatenate([c, dist_sq[mask] - r_sum[mask]**2])
    return c

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_v = None
    
    # Phase 1: Diverse initializations to locate promising basins
    for seed in range(15):
        np.random.seed(seed)
        centers = []
        
        if seed < 5:
            # Hexagonal lattice initialization (dense packing heuristic)
            r_est = 0.085
            y = r_est
            row = 0
            while len(centers) < N + 5:
                x_start = r_est + (row % 2) * r_est
                x = x_start
                while x <= 1 - r_est and len(centers) < N + 5:
                    centers.append([x, y])
                    x += 2 * r_est
                y += r_est * np.sqrt(3)
                row += 1
        else:
            # Random initialization with boundary padding
            centers = np.random.uniform(0.1, 0.9, size=(N, 2)).tolist()
            
        centers = np.array(centers[:N])
        # Add controlled jitter to break symmetry
        centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
        
        r_init = np.full(N, 0.02)  # Start small to guarantee initial feasibility
        v0 = np.concatenate([centers.flatten(), r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
            # Accept if feasible and improves best sum
            if np.all(constraints(res.x) >= -1e-8):
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement on the best solution found
    if best_v is not None:
        for _ in range(8):
            v_pert = best_v + np.random.normal(0, 0.002, size=best_v.shape)
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
            v_pert[2*N:] = np.clip(v_pert[2*N:], 0.001, 0.45)
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
                if np.all(constraints(res.x) >= -1e-8):
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_v = res.x.copy()
            except Exception:
                continue
                
    centers = best_v[:2*N].reshape(N, 2)
    radii = best_v[2*N:]
    
    # Safety check and minimal scaling to guarantee strict validity
    min_slack = 1.0
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        s = min(x - r, 1.0 - x - r, y - r, 1.0 - y - r)
        if s < min_slack: min_slack = s
        
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            s = dist - radii[i] - radii[j]
            if s < min_slack: min_slack = s
            
    # Apply minimal global shrink only if constraints are violated beyond tolerance
    if min_slack < -1e-9:
        scale = 1.0 + min_slack / np.max(radii)
        radii *= max(0.0, scale)
        
    return centers, radii, float(np.sum(radii))
