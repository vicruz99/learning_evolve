# sol_000032 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f550adc) state=4e3d1807 sum of radii=2.571023 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
I_IDX, J_IDX = np.triu_indices(N_CIRCLES, k=1)

def get_state(theta):
    """Transform optimization variables to physical centers and radii."""
    r = theta[:N_CIRCLES]
    u = theta[N_CIRCLES:2*N_CIRCLES]
    v = theta[2*N_CIRCLES:3*N_CIRCLES]
    # Transformation ensures r <= x,y <= 1-r automatically
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return x, y, r

def compute_obj(theta, mu):
    """Objective: minimize -sum(r) + penalty for overlaps."""
    x, y, r = get_state(theta)
    obj = -np.sum(r)
    
    # Vectorized pairwise distance and overlap computation
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    dr = r[I_IDX] + r[J_IDX]
    
    dist = np.sqrt(dx*dx + dy*dy)
    overlap = dr - dist
    overlap = np.maximum(0.0, overlap)
    
    obj += mu * np.sum(overlap * overlap)
    return obj

def run_packing():
    n = N_CIRCLES
    bounds = [(1e-5, 0.5)] * n + [(0.0, 1.0)] * (2 * n)
    
    best_theta = None
    best_sum_r = -np.inf
    
    # Multiple restarts to improve global search
    for restart in range(5):
        np.random.seed(restart * 42 + 123)
        
        # Initialize on a perturbed grid
        y_base = np.linspace(0.2, 0.8, 5)
        x_base = np.linspace(0.2, 0.8, 6)
        y_base += np.random.uniform(-0.03, 0.03, size=5)
        x_base += np.random.uniform(-0.03, 0.03, size=6)
        
        positions = []
        for iy in y_base:
            for ix in x_base:
                positions.append([ix, iy])
                if len(positions) == n:
                    break
            if len(positions) == n:
                break
        positions = np.array(positions)
        
        r_init = 0.08 + np.random.uniform(-0.005, 0.005, n)
        r_init = np.clip(r_init, 0.05, 0.15)
        
        x_init = positions[:, 0]
        y_init = positions[:, 1]
        
        # Solve for u, v parameters
        denom = 1.0 - 2.0 * r_init
        u_init = (x_init - r_init) / denom
        v_init = (y_init - r_init) / denom
        u_init = np.clip(u_init, 0.0, 1.0)
        v_init = np.clip(v_init, 0.0, 1.0)
        
        theta0 = np.concatenate([r_init, u_init, v_init])
        
        # Continuation method: gradually increase penalty weight
        mu = 100.0
        current_theta = theta0.copy()
        for step in range(8):
            res = minimize(compute_obj, current_theta, args=(mu,), method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 1500, 'ftol': 1e-11})
            current_theta = res.x
            mu *= 5.0
            
        x, y, r = get_state(current_theta)
        sum_r = np.sum(r)
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_theta = current_theta.copy()
            
    x, y, r = get_state(best_theta)
    centers = np.column_stack([x, y])
    
    # Post-processing: strictly enforce non-overlap constraint
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    dist = np.sqrt(dx*dx + dy*dy)
    req_dist = r[I_IDX] + r[J_IDX]
    overlap = np.maximum(0.0, req_dist - dist)
    max_overlap = np.max(overlap) if len(overlap) > 0 else 0.0
    
    if max_overlap > 1e-9:
        # Scale radii down proportionally to resolve residual overlaps
        ratios = dist / req_dist
        scale = np.min(ratios)
        r = r * scale
        
    return centers, r, float(np.sum(r))
