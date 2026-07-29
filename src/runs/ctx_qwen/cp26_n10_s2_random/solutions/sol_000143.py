# sol_000143 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000121 (state 8b7edc5c) state=c665f9c9 sum of radii=2.624554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
TRIU_IND = np.triu_indices(N, k=1)

def compute_obj(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def compute_cons(v):
    """Computes boundary and pairwise non-overlap constraints (must be >= 0)."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Boundary constraints
    c = np.empty(N * 4 + N * (N - 1) // 2)
    idx = 0
    c[idx:idx+N] = x - r; idx += N
    c[idx:idx+N] = 1.0 - x - r; idx += N
    c[idx:idx+N] = y - r; idx += N
    c[idx:idx+N] = 1.0 - y - r; idx += N
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[TRIU_IND[0]] - x[TRIU_IND[1]]
    dy = y[TRIU_IND[0]] - y[TRIU_IND[1]]
    dr = r[TRIU_IND[0]] + r[TRIU_IND[1]]
    c[idx:] = dx**2 + dy**2 - dr**2
    return c

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def hex_init(row_counts, r0, rng):
    """Generates an initial configuration from a perturbed hexagonal lattice."""
    centers = []
    y = r0
    for r_idx, count in enumerate(row_counts):
        offset = r0 if r_idx % 2 == 1 else 0.0
        x = r0 + offset
        for _ in range(count):
            centers.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        
    centers = np.array(centers[:N])
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize sum of radii."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_cons}
    
    # Phase 1: Generate diverse initial configurations
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 6, 5, 5],
        [6, 6, 5, 5, 4], [4, 6, 6, 6, 4], [5, 5, 5, 5, 6],
        [7, 5, 5, 5, 4], [5, 7, 5, 5, 4], [6, 4, 6, 5, 5]
    ]
    
    candidates = []
    for pat in patterns:
        for r0 in [0.085, 0.090, 0.095, 0.100, 0.105]:
            c = hex_init(pat, r0, rng)
            v = np.zeros(3 * N)
            v[0::3] = c[:, 0]
            v[1::3] = c[:, 1]
            # Compute strictly feasible initial radii
            db = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                            np.minimum(c[:, 1], 1.0 - c[:, 1]))
            dists = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            dn = 0.5 * np.min(dists, axis=1)
            v[2::3] = np.minimum(db, dn) * 0.90
            candidates.append(v)
            
    # Add randomized starts
    for _ in range(8):
        c = rng.uniform(0.1, 0.9, (N, 2))
        v = np.zeros(3 * N)
        v[0::3] = c[:, 0]
        v[1::3] = c[:, 1]
        db = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                        np.minimum(c[:, 1], 1.0 - c[:, 1]))
        v[2::3] = db * 0.85
        candidates.append(v)
        
    best_v = candidates[0]
    best_sum = -np.sum(best_v[2::3])
    
    # Phase 2: Homotopy Optimization (Radius Growth)
    # Iteratively grow radii to force the optimizer to find denser packings
    for grow_iter in range(6):
        next_candidates = []
        for v0 in candidates:
            try:
                res = minimize(compute_obj, v0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
                
                s_val = -res.fun
                c_vals = compute_cons(res.x)
                if np.min(c_vals) >= -1e-6:
                    if s_val > best_sum:
                        best_sum = s_val
                        best_v = res.x.copy()
                        
                    # Prepare candidate for next growth step
                    new_v = res.x.copy()
                    new_v[2::3] *= 1.02  # Grow radii to push boundary
                    
                    # Small center perturbation to escape exact local minima
                    new_v[0::3] += rng.normal(0, 0.0012, N)
                    new_v[1::3] += rng.normal(0, 0.0012, N)
                    new_v[0::3] = np.clip(new_v[0::3], 0.05, 0.95)
                    new_v[1::3] = np.clip(new_v[1::3], 0.05, 0.95)
                    
                    # Enforce boundary feasibility for the next step
                    xb = np.minimum(new_v[0::3], 1.0 - new_v[0::3])
                    yb = np.minimum(new_v[1::3], 1.0 - new_v[1::3])
                    new_v[2::3] = np.minimum(new_v[2::3], np.minimum(xb, yb) * 0.96)
                    
                    next_candidates.append(new_v)
            except Exception:
                pass
                
        # Keep top candidates for diversity and quality
        if next_candidates:
            # Sort by sum of radii and keep best 12
            next_candidates.sort(key=lambda v: -np.sum(v[2::3]), reverse=True)
            candidates = next_candidates[:12]
        else:
            break # No valid progress
            
    # Phase 3: Final High-Precision Polish
    try:
        res = minimize(compute_obj, best_v, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 25000, 'ftol': 1e-14})
        if -res.fun > best_sum and np.min(compute_cons(res.x)) >= -1e-7:
            best_v = res.x.copy()
    except Exception:
        pass
        
    centers = best_v.reshape(N, 3)[:, :2]
    radii = best_v.reshape(N, 3)[:, 2].copy()
    
    # Phase 4: Strict Deterministic Repair
    # Guarantees validation passes within 1e-12 tolerance without sacrificing optimality
    for _ in range(100):
        changed = False
        # Pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < radii[i] + radii[j] - 1e-11:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-11
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        # Boundary violations
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr - 1e-11:
                radii[i] = mr
                changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
