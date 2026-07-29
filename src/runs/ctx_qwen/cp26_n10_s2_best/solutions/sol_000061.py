# sol_000061 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state fea4b3d4) state=e896cca2 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraints(v, n):
    """
    Computes inequality constraints.
    Returns a flattened array where each element must be >= 0.
    Uses vectorized operations for efficiency and stability.
    """
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    
    c_list = []
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c_list.append(x - r)
    c_list.append(1.0 - x - r)
    c_list.append(y - r)
    c_list.append(1.0 - y - r)
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Broadcasting creates N x N matrices
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dr = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Squared Euclidean distance minus squared sum of radii
    d2_minus_r2 = dx**2 + dy**2 - dr**2
    
    # Extract upper triangle (i < j) to avoid duplicates and self-constraints
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c_list.append(d2_minus_r2[mask])
    
    return np.concatenate(c_list)

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    best_sum = -1.0
    best_v = None
    
    # Run multiple restarts with varied initial configurations to escape local minima
    num_restarts = 12
    for seed in range(num_restarts):
        np.random.seed(seed)
        
        # Cycle through initialization strategies
        init_type = ['hex', 'grid', 'rand'][seed % 3]
        
        if init_type == 'hex':
            # Hexagonal lattice initialization
            r0 = 0.085 + np.random.uniform(-0.02, 0.02)
            centers = []
            y = r0
            row = 0
            while len(centers) < n + 5:
                x_start = r0 if row % 2 == 0 else 2 * r0
                x = x_start
                while x <= 1 - r0:
                    centers.append([x, y])
                    x += 2 * r0
                y += np.sqrt(3) * r0
                row += 1
            centers = np.array(centers[:n])
        elif init_type == 'grid':
            # Grid lattice initialization
            r0 = 0.085 + np.random.uniform(-0.02, 0.02)
            xs = np.linspace(r0, 1 - r0, 6)
            ys = np.linspace(r0, 1 - r0, 5)
            centers = []
            for cy in ys:
                for cx in xs:
                    centers.append([cx, cy])
                    if len(centers) >= n: break
                if len(centers) >= n: break
            centers = np.array(centers[:n])
        else:
            # Random initialization
            centers = np.random.uniform(0.1, 0.9, (n, 2))
            
        # Add controlled jitter to break symmetry and help exploration
        centers += np.random.uniform(-0.015, 0.015, centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
        
        # Start with small radii to ensure initial feasibility
        r_init = np.full(n, 0.04)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints, 'args': (n,)},
                           options={'maxiter': 4000, 'ftol': 1e-11, 'disp': False})
            
            if res.success or (res.x is not None):
                current_sum = -res.fun
                # Quick feasibility check before accepting
                if np.all(constraints(res.x, n) >= -1e-5):
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization fails completely
    if best_v is None:
        best_v = np.zeros(3*n)
        best_v[:n] = 0.5
        best_v[n:2*n] = 0.5
        best_v[2*n:] = 0.01
        
    x = best_v[:n]
    y = best_v[n:2*n]
    r = best_v[2*n:]
    centers = np.column_stack((x, y))
    radii = r.copy()
    
    # Post-processing to guarantee strict validity per validator tolerances
    # 1. Enforce boundary constraints strictly
    for i in range(n):
        radii[i] = min(radii[i], centers[i, 0], 1 - centers[i, 0], 
                       centers[i, 1], 1 - centers[i, 1])
        
    # 2. Enforce non-overlap constraints strictly with safety margin
    # Iteratively shrink overlapping pairs until valid
    for _ in range(30):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                
                # If overlap exists beyond numerical tolerance, shrink both equally
                if dist < sum_r - 1e-13:
                    shrink = (sum_r - dist + 1e-8) / 2.0
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
