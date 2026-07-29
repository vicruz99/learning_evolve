# sol_000090 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000068 (state 22e68fa8) state=3b7e6ace sum of radii=2.628435 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective_func(v):
    """Objective: minimize negative sum of radii to maximize total radius."""
    return -np.sum(v[2::3])

def compute_constraints(v):
    """Computes boundary and pairwise non-overlap constraints (must be >= 0)."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c = np.concatenate([c, (dx**2 + dy**2 - dr**2)[mask]])
    return c

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def hex_init(row_counts, r0, seed):
    """Generates initial configuration from a perturbed hexagonal lattice."""
    rng = np.random.default_rng(seed)
    centers = []
    y = r0
    for row_idx, count in enumerate(row_counts):
        offset = r0 if row_idx % 2 == 1 else 0.0
        x = r0 + offset
        for _ in range(count):
            centers.append([x + rng.normal(0, 0.005), y + rng.normal(0, 0.005)])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
    centers = np.array(centers[:N])
    centers = np.clip(centers, 0.05, 0.95)
    return centers

def repair_solution(centers, radii):
    """Iteratively shrinks radii to resolve overlaps and clamp to boundaries."""
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                if d < radii[i] + radii[j] - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        for i in range(N):
            mr = min(centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
            if radii[i] > mr - 1e-9:
                radii[i] = mr
                changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    return radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Main function to pack 26 circles in a unit square."""
    rng = np.random.default_rng(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_v = None
    best_sum = -np.inf
    
    # Phase 1: Generate diverse initial configurations
    row_patterns = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6], [6, 5, 5, 5, 5], [5, 5, 5, 6, 5]]
    inits = []
    
    for pat in row_patterns:
        for r0 in [0.09, 0.095, 0.10, 0.105]:
            c = hex_init(pat, r0, len(inits))
            v = np.zeros(3 * N)
            v[0::3] = c[:, 0]
            v[1::3] = c[:, 1]
            
            # Compute feasible initial radii based on distances to boundaries and neighbors
            rs = np.full(N, 0.05)
            for i in range(N):
                db = min(c[i,0], 1.0 - c[i,0], c[i,1], 1.0 - c[i,1])
                mask = np.ones(N, dtype=bool)
                mask[i] = False
                dn = np.min(np.hypot(c[i,0] - c[mask,0], c[i,1] - c[mask,1]))
                rs[i] = min(db, dn / 2.0) * 0.9
            v[2::3] = rs
            inits.append(v)
            
    # Add random starts to explore non-lattice optima
    for _ in range(20):
        c = rng.random((N, 2)) * 0.8 + 0.1
        v = np.zeros(3 * N)
        v[0::3] = c[:, 0]
        v[1::3] = c[:, 1]
        v[2::3] = 0.04
        inits.append(v)
        
    # Phase 2: Multi-start optimization
    for i, v0 in enumerate(inits):
        try:
            res = minimize(objective_func, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 20000, 'ftol': 1e-13})
            s_val = -res.fun
            if s_val > best_sum:
                c_vals = compute_constraints(res.x)
                if np.min(c_vals) >= -1e-7:
                    best_sum = s_val
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Local perturbation search to escape local minima
    if best_v is not None:
        for _ in range(60):
            v_trial = best_v + rng.normal(0, 0.0012, best_v.shape)
            v_trial[0::3] = np.clip(v_trial[0::3], 0.02, 0.98)
            v_trial[1::3] = np.clip(v_trial[1::3], 0.02, 0.98)
            v_trial[2::3] = np.clip(v_trial[2::3], 0.01, 0.45)
            try:
                res = minimize(objective_func, v_trial, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13})
                if -res.fun > best_sum:
                    c_vals = compute_constraints(res.x)
                    if np.min(c_vals) >= -1e-7:
                        best_sum = -res.fun
                        best_v = res.x.copy()
            except Exception:
                pass
                
    # Phase 4: Extract and strictly repair solution
    centers = np.column_stack((best_v[0::3], best_v[1::3]))
    radii = best_v[2::3].copy()
    radii = repair_solution(centers, radii)
    
    return centers, radii, float(np.sum(radii))
