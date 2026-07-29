# sol_000121 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000090 (state 3b7e6ace) state=8b7edc5c sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pairwise interaction indices for faster constraint evaluation
MASK = np.triu_indices(N, k=1)

def compute_obj(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def compute_cons(v):
    """Computes boundary and pairwise non-overlap constraints (must be >= 0)."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    return np.concatenate([c, (dx**2 + dy**2 - dr**2)[MASK]])

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def make_init_hex(row_counts, r0, rng):
    """Generates an initial configuration from a perturbed hexagonal lattice."""
    centers = []
    y = r0
    for row_idx, count in enumerate(row_counts):
        offset = r0 if row_idx % 2 == 1 else 0.0
        x = r0 + offset
        for _ in range(count):
            centers.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        
    centers = np.array(centers[:N])
    centers += rng.normal(0, 0.004, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize sum of radii."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_cons}
    rng = np.random.default_rng(42)
    
    best_v = None
    best_sum = -1e9
    
    # Phase 1: Generate diverse initial configurations
    pool = []
    row_patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], 
        [6, 5, 5, 5, 5], [5, 7, 5, 5, 4], [4, 6, 6, 6, 4],
        [6, 4, 6, 5, 5], [5, 5, 6, 5, 5]
    ]
    
    for pat in row_patterns:
        for r0 in [0.09, 0.095, 0.10, 0.105, 0.11]:
            c = make_init_hex(pat, r0, rng)
            v = np.zeros(3 * N)
            v[0::3] = c[:, 0]
            v[1::3] = c[:, 1]
            
            # Compute strictly feasible initial radii
            rs = np.full(N, 0.05)
            for i in range(N):
                db = min(c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
                mask = np.ones(N, dtype=bool)
                mask[i] = False
                dn = np.min(np.hypot(c[i, 0] - c[mask, 0], c[i, 1] - c[mask, 1]))
                rs[i] = min(db, dn / 2.0) * 0.95
            v[2::3] = rs
            pool.append(v)
            
    current_v = pool[0]
    
    # Phase 2: Iterative Optimization with Adaptive Perturbation
    for cycle in range(50):
        # Decide whether to perturb the best found or try a fresh start
        if best_v is not None and rng.random() < 0.75:
            v0 = best_v.copy()
            # Decaying noise schedule
            scale = 0.006 * (0.96 ** cycle)
            v0[0::3] += rng.normal(0, scale, N)
            v0[1::3] += rng.normal(0, scale, N)
            v0[0::3] = np.clip(v0[0::3], 0.02, 0.98)
            v0[1::3] = np.clip(v0[1::3], 0.02, 0.98)
            # Shrink radii slightly to create room for repositioning
            v0[2::3] *= 0.95
        else:
            v0 = pool.pop(0) if pool else best_v.copy()
            
        try:
            res = minimize(compute_obj, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            
            s_val = -res.fun
            if s_val > best_sum:
                # Verify constraint satisfaction (allow tiny numerical slack)
                c_vals = compute_cons(res.x)
                if np.min(c_vals) >= -1e-6:
                    best_sum = s_val
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Extract and Strictly Repair Solution
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = best_v[2::3].copy()
    
    # Deterministic repair to guarantee validation passes within 1e-12 tolerance
    for _ in range(100):
        changed = False
        # Resolve pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-10
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        # Resolve boundary violations
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr - 1e-10:
                radii[i] = mr
                changed = True
                
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
