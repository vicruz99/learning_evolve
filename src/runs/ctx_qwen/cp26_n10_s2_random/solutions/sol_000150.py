# sol_000150 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000141 (state d8f6c168) state=127c3a4f sum of radii=2.625767 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

# Precompute pairwise indices for efficient constraint evaluation
PAIR_IDX = np.triu_indices(N, k=1)

def objective_func(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraint_func(v):
    """Computes boundary and non-overlap constraints. Must return >= 0."""
    centers = v[:2*N].reshape(N, 2)
    radii = v[2*N:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])
    
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    c_i = centers[PAIR_IDX[0]]
    c_j = centers[PAIR_IDX[1]]
    r_i = radii[PAIR_IDX[0]]
    r_j = radii[PAIR_IDX[1]]
    dists = np.linalg.norm(c_i - c_j, axis=1)
    con = np.concatenate([con, dists - (r_i + r_j)])
    
    return con

def get_variable_bounds():
    """Returns bounds for x, y in [0,1] and r in [0, 0.5]."""
    return [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

def generate_hex_init(pattern, r0, rng):
    """Generates a perturbed hexagonal lattice configuration."""
    centers = []
    y = r0
    for r_idx, count in enumerate(pattern):
        shift = r0 if r_idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(count):
            centers.append([x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)])
            x += 2.0 * r0
        y += r0 * np.sqrt(3.0)
        
    centers = np.array(centers[:N])
    centers = np.clip(centers, 0.05, 0.95)
    
    v = np.zeros(3 * N)
    v[:2*N] = centers.flatten()
    v[2*N:] = np.full(N, r0 * 0.85)
    return v

def strict_repair(centers, radii):
    """Iteratively adjusts radii to strictly satisfy validator tolerances."""
    for _ in range(150):
        changed = False
        # Fix pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2.0 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
                    
        # Fix boundary violations
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max_r
                changed = True
                
        if not changed:
            break
            
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds = get_variable_bounds()
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_v = None
    best_sum = -np.inf
    
    # Phase 1: Generate diverse structured and random starts
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 5, 5, 6],
        [6, 4, 6, 5, 5], [4, 6, 6, 6, 4], [5, 5, 6, 5, 5],
        [5, 4, 6, 6, 5], [6, 6, 5, 5, 4]
    ]
    
    starts = []
    for pat in patterns:
        for r0 in [0.088, 0.092, 0.096, 0.100, 0.104]:
            starts.append(generate_hex_init(pat, r0, rng))
            
    for _ in range(20):
        c = rng.uniform(0.12, 0.88, (N, 2))
        v = np.zeros(3 * N)
        v[:2*N] = c.flatten()
        v[2*N:] = 0.06
        starts.append(v)
        
    # Phase 2: Multi-start constrained optimization
    for v0 in starts:
        try:
            res = minimize(objective_func, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                c_vals = constraint_func(res.x)
                if np.min(c_vals) >= -1e-7:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Adaptive perturbation search to escape local minima
    if best_v is not None:
        for step in range(60):
            # Cooling schedule: start with larger perturbations, decay to fine-tuning
            sigma = 0.006 * (1.0 - step / 60.0) + 0.0005
            v_trial = best_v + rng.normal(0, sigma, best_v.shape)
            v_trial[:2*N] = np.clip(v_trial[:2*N], 0.02, 0.98)
            v_trial[2*N:] = np.clip(v_trial[2*N:], 0.01, 0.45)
            
            try:
                res = minimize(objective_func, v_trial, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-13, 'disp': False})
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    c_vals = constraint_func(res.x)
                    if np.min(c_vals) >= -1e-7:
                        best_sum = curr_sum
                        best_v = res.x.copy()
            except Exception:
                pass
                
    # Phase 4: Extract and strictly repair for validator compliance
    if best_v is None:
        best_v = starts[0]
        
    centers = best_v[:2*N].reshape(N, 2)
    radii = best_v[2*N:].copy()
    radii = strict_repair(centers, radii)
    
    return centers, radii, float(np.sum(radii))
