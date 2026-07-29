# sol_000120 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000083 (state c6ee3a07) state=f5aab9f3 sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute indices for all unique circle pairs to speed up constraint evaluation
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2*N:])

def constraints(v):
    """
    Computes all inequality constraints.
    Returns a flat array where each element >= 0 indicates a satisfied constraint.
    Uses squared distances for better numerical stability.
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist_sq = dx**2 + dy**2
    r_sum_sq = (r[PAIR_I] + r[PAIR_J])**2
    c = np.concatenate([c, dist_sq - r_sum_sq])
    
    return c

def ensure_feasible(v):
    """Adjusts radii to guarantee the configuration satisfies all constraints strictly."""
    x = v[:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:].copy()
    
    # Enforce boundary constraints
    r = np.minimum(r, x)
    r = np.minimum(r, 1.0 - x)
    r = np.minimum(r, y)
    r = np.minimum(r, 1.0 - y)
    
    # Enforce non-overlap constraints iteratively
    for _ in range(20):
        dx = x[PAIR_I] - x[PAIR_J]
        dy = y[PAIR_I] - y[PAIR_J]
        dist = np.sqrt(dx**2 + dy**2)
        overlap = (r[PAIR_I] + r[PAIR_J]) - dist
        
        if np.max(overlap) < 1e-9:
            break
            
        # Shrink both circles equally to resolve overlap, add tiny buffer
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-9
        r[PAIR_I] = np.maximum(0.0, r[PAIR_I] - shrink)
        r[PAIR_J] = np.maximum(0.0, r[PAIR_J] - shrink)
        
    return np.concatenate([x, y, r])

def run_packing():
    # Bounds: coordinates in [0, 1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # 1. Generate diverse initial configurations
    initial_centers = []
    
    # Hexagonal lattice
    r_hex = 0.10
    pts = []
    y = r_hex
    row = 0
    while len(pts) < N + 10:
        x = r_hex if row % 2 == 0 else 2 * r_hex
        while x <= 1.0 - r_hex:
            pts.append([x, y])
            x += 2 * r_hex
        y += r_hex * np.sqrt(3)
        row += 1
    initial_centers.append(np.array(pts[:N]))
    
    # Rotated hexagonal lattices (often better for boundary fitting)
    base_pts = np.array(pts[:N+10])
    for angle in [0.15, 0.3, 0.5]:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = base_pts - 0.5
        rotated = rotated @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]) + 0.5
        valid = (rotated[:,0] > 0.05) & (rotated[:,0] < 0.95) & \
                (rotated[:,1] > 0.05) & (rotated[:,1] < 0.95)
        if np.sum(valid) >= N:
            initial_centers.append(rotated[valid][:N])
            
    # Random uniform placements
    for seed in range(5):
        np.random.seed(seed)
        initial_centers.append(np.random.uniform(0.1, 0.9, (N, 2)))
        
    # Grid based
    gx = np.linspace(0.1, 0.9, 6)
    gy = np.linspace(0.1, 0.9, 5)
    cx, cy = np.meshgrid(gx, gy)
    initial_centers.append(np.column_stack([cx.flatten(), cy.flatten()])[:N])
    
    # Corner-focused placements
    for seed in range(3):
        np.random.seed(100 + seed)
        c_corner = np.array([[0.12,0.12], [0.88,0.12], [0.12,0.88], [0.88,0.88]])
        c_rest = np.random.uniform(0.2, 0.8, (N-4, 2))
        initial_centers.append(np.vstack([c_corner, c_rest]))
        
    # 2. Primary Multi-start Optimization
    for base_c in initial_centers:
        for seed in range(8):
            np.random.seed(seed)
            pert_c = base_c + np.random.uniform(-0.015, 0.015, base_c.shape)
            pert_c = np.clip(pert_c, 0.03, 0.97)
            
            v0 = np.zeros(3*N)
            v0[:N] = pert_c[:, 0]
            v0[N:2*N] = pert_c[:, 1]
            v0[2*N:] = 0.07  # Initial radii
            
            v0 = ensure_feasible(v0)
            
            try:
                res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-11})
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    # Verify feasibility before accepting
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-6:
                        best_sum = curr_sum
                        best_v = res.x.copy()
            except Exception:
                pass
                
    # 3. Perturbation & Refinement Loop to escape local minima
    if best_v is not None:
        for step in range(20):
            np.random.seed(step + 1000)
            v_curr = best_v.copy()
            
            # Shrink radii to create breathing room for center movement
            scale = 0.96 - step * 0.002
            v_curr[2*N:] *= max(0.9, scale)
            
            # Perturb centers
            noise = np.random.uniform(-0.008, 0.008, 2*N)
            v_curr[:2*N] += noise
            v_curr[:2*N] = np.clip(v_curr[:2*N], 0.02, 0.98)
            
            v_curr = ensure_feasible(v_curr)
            
            try:
                res = minimize(objective, v_curr, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-11})
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-6:
                        best_sum = curr_sum
                        best_v = res.x.copy()
            except Exception:
                pass
                
    # 4. Fallback if optimization completely fails (highly unlikely)
    if best_v is None:
        centers = np.random.uniform(0.2, 0.8, (N, 2))
        radii = np.full(N, 0.04)
        best_v = np.concatenate([centers[:,0], centers[:,1], radii])
        best_v = ensure_feasible(best_v)
        
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Strict post-processing to guarantee validation passes
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, centers[:,0])
    radii = np.minimum(radii, 1.0 - centers[:,0])
    radii = np.minimum(radii, centers[:,1])
    radii = np.minimum(radii, 1.0 - centers[:,1])
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(15):
        dx = centers[PAIR_I, 0] - centers[PAIR_J, 0]
        dy = centers[PAIR_I, 1] - centers[PAIR_J, 1]
        dist = np.sqrt(dx**2 + dy**2)
        sum_r = radii[PAIR_I] + radii[PAIR_J]
        overlap = sum_r - dist
        
        if np.max(overlap) < 1e-9:
            break
            
        shrink = np.maximum(0.0, overlap) / 2.0 + 1e-9
        radii[PAIR_I] = np.maximum(0.0, radii[PAIR_I] - shrink)
        radii[PAIR_J] = np.maximum(0.0, radii[PAIR_J] - shrink)
        
    return centers, radii, float(np.sum(radii))
