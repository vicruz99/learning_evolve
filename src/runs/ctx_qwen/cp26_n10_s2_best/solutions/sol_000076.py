# sol_000076 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000001 (state 1501c8b5) state=ccc88021 sum of radii=2.579206 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def constraint_func_sq(v, n):
    """
    Computes inequality constraints using squared distances for numerical stability.
    Returns a flat array where all elements must be >= 0.
    """
    centers = v[:2*n].reshape(n, 2)
    radii = v[2*n:]
    
    cons = []
    # Boundary constraints
    cons.append(centers[:, 0] - radii)
    cons.append(1 - centers[:, 0] - radii)
    cons.append(centers[:, 1] - radii)
    cons.append(1 - centers[:, 1] - radii)
    
    # Pairwise non-overlap constraints (squared)
    c1 = centers[:, np.newaxis, :]
    c2 = centers[np.newaxis, :, :]
    dists_sq = np.sum((c1 - c2)**2, axis=2)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append(dists_sq[mask] - r_sum[mask]**2)
    
    return np.concatenate(cons)

def objective(v, n):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2*n:])

def generate_hex_init(n, r_init=0.06):
    """Generates a hexagonal lattice initialization."""
    centers = []
    y = r_init
    row = 0
    while len(centers) < n:
        x_start = r_init + (row % 2) * r_init
        x = x_start
        while x <= 1 - r_init and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_init
        y += r_init * np.sqrt(3)
        row += 1
    return np.array(centers[:n])

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    configs = []
    
    # 1. Hexagonal lattices with varying base radii
    for r in [0.05, 0.06, 0.07]:
        configs.append((generate_hex_init(n, r), np.full(n, 0.05)))
        
    # 2. Grid lattices
    for spacing in [0.15, 0.12, 0.10]:
        pts = []
        y = spacing
        while y <= 1 - spacing:
            x = spacing
            while x <= 1 - spacing:
                pts.append([x, y])
                x += spacing
            y += spacing
        pts = np.array(pts[:n])
        configs.append((pts, np.full(n, 0.05)))
        
    # 3. Random starts
    for seed in range(8):
        np.random.seed(seed)
        pts = np.random.uniform(0.1, 0.9, size=(n, 2))
        configs.append((pts, np.full(n, 0.04)))
        
    # 4. Perturbed hexagonal starts
    for seed in range(5):
        np.random.seed(100+seed)
        pts = generate_hex_init(n, 0.06) + np.random.uniform(-0.02, 0.02, size=(n,2))
        pts = np.clip(pts, 0.05, 0.95)
        configs.append((pts, np.full(n, 0.05)))

    # Multi-start optimization
    for i, (init_centers, init_radii) in enumerate(configs):
        x0 = np.concatenate([init_centers.flatten(), init_radii])
        
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraint_func_sq, 'args': (n,)},
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            
            current_sum = -res.fun
            if current_sum > best_sum:
                cons_vals = constraint_func_sq(res.x, n)
                # Accept if sufficiently feasible
                if np.all(cons_vals >= -1e-5):
                    best_sum = current_sum
                    best_centers = res.x[:2*n].reshape(n, 2).copy()
                    best_radii = res.x[2*n:].copy()
        except Exception:
            continue
            
    # Polishing phase: perturb best found and re-optimize
    if best_centers is not None:
        for _ in range(5):
            np.random.seed(_)
            noisy_centers = best_centers + np.random.uniform(-0.005, 0.005, size=best_centers.shape)
            noisy_centers = np.clip(noisy_centers, 0.01, 0.99)
            x0 = np.concatenate([noisy_centers.flatten(), best_radii])
            try:
                res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                               constraints={'type': 'ineq', 'fun': constraint_func_sq, 'args': (n,)},
                               options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                current_sum = -res.fun
                if current_sum > best_sum:
                    cons_vals = constraint_func_sq(res.x, n)
                    if np.all(cons_vals >= -1e-5):
                        best_sum = current_sum
                        best_centers = res.x[:2*n].reshape(n, 2).copy()
                        best_radii = res.x[2*n:].copy()
            except Exception:
                pass

    # Fallback if optimization completely fails
    if best_centers is None:
        centers = generate_hex_init(n, 0.05)
        radii = np.full(n, 0.05)
    else:
        centers = best_centers
        radii = best_radii
        
    # Strict post-processing to guarantee validity within checker tolerance
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1-x, y, 1-y)
        
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = radii[i] + radii[j]
            if dist < sum_r - 1e-9:
                shrink = (sum_r - dist) / 2.0 + 1e-7
                radii[i] = max(0.0, radii[i] - shrink)
                radii[j] = max(0.0, radii[j] - shrink)
                
    return centers, radii, float(np.sum(radii))
