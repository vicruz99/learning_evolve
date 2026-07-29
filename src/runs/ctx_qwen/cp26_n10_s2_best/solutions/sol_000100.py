# sol_000100 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state af044a19) state=9b4c1d1c sum of radii=2.628083 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(v, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraint_func(v, n, pair_i, pair_j):
    """Compute inequality constraints: boundaries and pairwise non-overlap."""
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    
    cons = []
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Pairwise non-overlap: dist >= r_i + r_j
    c_i = c[pair_i]
    c_j = c[pair_j]
    dists = np.sqrt(np.sum((c_i - c_j)**2, axis=1))
    r_sum = r[pair_i] + r[pair_j]
    cons.append(dists - r_sum)
    
    return np.concatenate(cons)

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pair_i, pair_j = np.triu_indices(n, k=1)
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraint_func, 'args': (n, pair_i, pair_j)}
    
    best_sum = -1.0
    best_v = None
    
    def try_optimize(v0):
        nonlocal best_sum, best_v
        try:
            res = minimize(objective_func, v0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_sum:
                # Strict feasibility check before accepting
                c_val = constraint_func(res.x, n, pair_i, pair_j)
                if np.all(c_val >= -1e-7):
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            pass

    # --- Phase 1: Diverse Initial Configurations ---
    configs = []
    
    # 1. Hexagonal lattices with shifts & base radii
    for r0 in [0.085, 0.095, 0.105, 0.115]:
        for sx in [-0.02, 0.0, 0.02, 0.04]:
            for sy in [-0.02, 0.0, 0.02]:
                pts = []
                y = r0 + sy
                row = 0
                while len(pts) < n + 5:
                    x_start = r0 + sx + (row % 2) * r0
                    x = x_start
                    while x <= 1.0 - r0 and len(pts) < n + 5:
                        pts.append([x, y])
                        x += 2 * r0
                    y += np.sqrt(3) * r0
                    row += 1
                configs.append(np.array(pts[:n]))
                
    # 2. Square grids
    for s in [0.05, 0.08, 0.12, 0.15]:
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([i*0.17 + s, j*0.19 + s])
        configs.append(np.array(pts[:n]))
        
    # 3. Random points relaxed with repulsive forces (Lloyd-like)
    np.random.seed(42)
    for k in range(15):
        pts = np.random.uniform(0.1, 0.9, (n, 2))
        for _ in range(60):
            forces = np.zeros_like(pts)
            for i in range(n):
                diff = pts[i] - pts
                dist = np.linalg.norm(diff, axis=1)
                dist[dist < 1e-6] = 1e-6
                rep = 1.0 / (dist**2)
                forces[i] += np.sum(diff * rep[:, None], axis=0)
            pts += 0.012 * forces
            pts = np.clip(pts, 0.05, 0.95)
        configs.append(pts)
        
    # 4. Corner-heavy configurations
    for seed in range(8):
        np.random.seed(1000 + seed)
        pts = np.array([[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]])
        pts = np.vstack([pts, np.random.uniform(0.2, 0.8, (n-4, 2))])
        configs.append(pts)
        
    # Run optimization on all configs
    for i, base in enumerate(configs):
        np.random.seed(i)
        c_pert = base + np.random.uniform(-0.006, 0.006, base.shape)
        c_pert = np.clip(c_pert, 0.05, 0.95)
        v0 = np.concatenate([c_pert.flatten(), np.full(n, 0.05)])
        try_optimize(v0)
        
    # --- Phase 2: Iterative Perturbation to Escape Local Minima ---
    if best_v is not None:
        for step in range(15):
            v_pert = best_v.copy()
            # Gradually shrink radii to create space for centers to move
            scale = 0.96 - step * 0.004
            v_pert[2*n:] *= scale
            # Perturb centers
            v_pert[:2*n] += np.random.uniform(-0.005, 0.005, 2*n)
            v_pert[:2*n] = np.clip(v_pert[:2*n], 0.05, 0.95)
            try_optimize(v_pert)
            
    # Fallback safety
    if best_v is None:
        best_v = np.zeros(3*n)
        best_v[:2*n] = np.random.uniform(0.2, 0.8, 2*n)
        best_v[2*n:] = 0.03
        
    centers = best_v[:2*n].reshape(n, 2)
    radii = best_v[2*n:]
    
    # --- Phase 3: Strict Post-Processing ---
    # Guarantee validator compliance by iteratively shrinking radii to resolve violations
    for _ in range(10):
        changed = False
        # Boundary constraints
        for i in range(n):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-9:
                radii[i] = mr
                changed = True
                
        # Pairwise non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d + 1e-9:
                    shr = (radii[i] + radii[j] - d) * 0.5 + 1e-7
                    radii[i] = max(0.0, radii[i] - shr)
                    radii[j] = max(0.0, radii[j] - shr)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
