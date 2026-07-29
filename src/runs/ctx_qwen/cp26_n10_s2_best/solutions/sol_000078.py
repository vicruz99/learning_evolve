# sol_000078 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000001 (state 1501c8b5) state=c7bef873 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective_func(v):
    """Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraint_func(v):
    """Compute inequality constraints: boundaries and pairwise non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    con_list = [
        x - r,
        1 - x - r,
        y - r,
        1 - y - r
    ]
    
    # Pairwise constraints: dist^2 >= (r_i + r_j)^2
    # Using squared distance avoids non-smoothness at dist=0 and improves optimizer stability
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    # Only consider upper triangle to avoid duplicate constraints
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    con_list.append(dist_sq[mask] - r_sum[mask]**2)
    
    return np.concatenate(con_list)

def run_packing():
    np.random.seed(42)
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    inits = []
    
    # 1. Hexagonal lattice patterns with varying row distributions
    # These approximate optimal dense packings for N=26
    row_patterns = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [6, 6, 5, 5, 4],
        [5, 5, 6, 6, 4],
        [6, 4, 6, 5, 5],
        [5, 6, 6, 4, 5],
        [6, 5, 5, 5, 5]
    ]
    
    for rows in row_patterns:
        est_r = 0.085  # Initial radius estimate allowing room for growth
        pts = []
        y = est_r
        for i, count in enumerate(rows):
            x_start = est_r + (est_r if i % 2 == 1 else 0)
            x = x_start
            for _ in range(count):
                if len(pts) < N:
                    pts.append([x, y])
                    x += 2 * est_r
            y += est_r * np.sqrt(3)
            
        pts = np.array(pts[:N])
        # Add perturbation to break symmetry and allow boundary fitting
        pts += np.random.uniform(-0.015, 0.015, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        r_init = np.full(N, est_r)
        inits.append(np.concatenate([pts.ravel(), r_init]))
        
    # 2. Structured grid variations
    for seed in range(5):
        np.random.seed(seed + 1000)
        grid_x = np.linspace(0.1, 0.9, 6)
        grid_y = np.linspace(0.1, 0.9, 5)
        pts = []
        for y in grid_y:
            for x in grid_x:
                pts.append([x, y])
        pts = np.array(pts[:N])
        pts += np.random.uniform(-0.02, 0.02, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        r_init = np.full(N, 0.07)
        inits.append(np.concatenate([pts.ravel(), r_init]))
        
    # 3. Random starts for diversity
    for seed in range(8):
        np.random.seed(seed + 2000)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        r_init = np.full(N, 0.04)
        inits.append(np.concatenate([pts.ravel(), r_init]))
        
    # Optimization loop
    for v0 in inits:
        try:
            res = minimize(objective_func, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
            val = -res.fun
            if val > best_sum:
                best_sum = val
                best_centers = res.x[:2*N].reshape(N, 2).copy()
                best_radii = res.x[2*N:].copy()
        except Exception:
            continue
            
    # Refinement phase: perturb best solution and re-optimize to escape local minima
    for _ in range(4):
        v_cur = np.concatenate([best_centers.ravel(), best_radii])
        noise = np.random.uniform(-1e-4, 1e-4, v_cur.shape)
        v_pert = v_cur + noise
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.01, 0.99)
        v_pert[2*N:] = np.maximum(v_pert[2*N:], 0.005)
        
        try:
            res = minimize(objective_func, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            val = -res.fun
            if val > best_sum:
                best_sum = val
                best_centers = res.x[:2*N].reshape(N, 2).copy()
                best_radii = res.x[2*N:].copy()
        except Exception:
            continue
            
    # Post-processing to guarantee strict validity per validation rules
    centers = best_centers
    radii = best_radii
    
    # Ensure boundaries are strictly satisfied
    for i in range(N):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1-x, y, 1-y)
        
    # Iteratively enforce non-overlap strictly
    for _ in range(5):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < radii[i] + radii[j]:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-8
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    # Final tiny scale down to safely clear the 1e-12 numerical tolerance in validator
    radii *= 0.999999999
            
    return centers, radii, float(np.sum(radii))
