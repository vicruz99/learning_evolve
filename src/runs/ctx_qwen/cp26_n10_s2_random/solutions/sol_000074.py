# sol_000074 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000054 (state 91c332d2) state=61c7f7c2 sum of radii=2.617047 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def get_feasible_radii(centers):
    """Computes the maximum valid radius for each circle given fixed centers."""
    x, y = centers[:, 0], centers[:, 1]
    # Distance to boundaries
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Pairwise distances
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Radius limited by half the distance to nearest neighbor
    r_pair = 0.5 * np.min(dists, axis=1)
    
    return np.minimum(r_bound, r_pair)

def objective_func(v):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2 * N:])

def constraint_func(v):
    """Computes all boundary and non-overlap constraints. Returns array >= 0 for feasibility."""
    centers = v[:2 * N].reshape(N, 2)
    radii = v[2 * N:]
    
    con = []
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con.append(centers[:, 0] - radii)
    con.append(1.0 - centers[:, 0] - radii)
    con.append(centers[:, 1] - radii)
    con.append(1.0 - centers[:, 1] - radii)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    idx = np.triu_indices(N, k=1)
    diffs = centers[idx[0]] - centers[idx[1]]
    dists = np.linalg.norm(diffs, axis=1)
    con.append(dists - (radii[idx[0]] + radii[idx[1]]))
    
    return np.concatenate(con)

def make_hex_init(rows_pattern, seed=42):
    """Generates a hexagonal lattice initialization with specified row structure."""
    rng = np.random.default_rng(seed)
    pts = []
    
    # Estimate spacing to fit circles comfortably
    r_est = 0.10
    dy = r_est * np.sqrt(3.0)
    dx = 2.0 * r_est
    
    y = r_est
    for r_idx, count in enumerate(rows_pattern):
        shift = (dx / 2.0) if (r_idx % 2 == 1) else 0.0
        x = r_est + shift
        for _ in range(count):
            pts.append([x + rng.uniform(-0.002, 0.002),
                        y + rng.uniform(-0.002, 0.002)])
            x += dx
        y += dy
        
    c = np.array(pts[:N])
    
    # Normalize and scale to fit inside [0.05, 0.95]
    c -= c.min(axis=0)
    c /= c.max(axis=0)
    c = c * 0.90 + 0.05
    
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-7, 0.5)] * N
    constraints = {'type': 'ineq', 'fun': constraint_func}
    
    best_v = None
    best_sum = -np.inf
    
    # Diverse structural patterns that sum to 26
    patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4],
        [6, 6, 6, 4, 4],
        [5, 5, 6, 5, 5]
    ]
    
    inits = []
    for p in patterns:
        inits.append(make_hex_init(p, seed=42))
        inits.append(make_hex_init(p, seed=123))
        
    # Add random dense starts
    np.random.seed(42)
    for _ in range(10):
        inits.append(np.random.rand(N, 2) * 0.8 + 0.1)
        
    # Phase 1: Global Multi-Start Optimization
    for c0 in inits:
        # Compute strictly feasible initial radii
        r0 = get_feasible_radii(c0) * 0.95
        r0 = np.maximum(r0, 1e-5)
        
        v0 = np.concatenate([c0.flatten(), r0])
        
        try:
            res = minimize(
                objective_func, v0, method='SLSQP', bounds=bounds,
                constraints=constraints, options={'maxiter': 10000, 'ftol': 1e-14}
            )
            
            curr_sum = -res.fun
            # Verify feasibility with tolerance
            c_vals = constraint_func(res.x)
            if np.min(c_vals) > -1e-7 and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization failed entirely
    if best_v is None:
        c_fallback = inits[0]
        r_fallback = get_feasible_radii(c_fallback)
        best_v = np.concatenate([c_fallback.flatten(), r_fallback])
        best_sum = np.sum(r_fallback)
        
    # Phase 2: Iterative Perturbation Search (Basin Hopping style)
    rng = np.random.default_rng(42)
    for step in range(35):
        noise_scale = 0.003 * (0.92 ** step)
        v_pert = best_v + rng.normal(0, noise_scale, best_v.shape)
        
        # Clip to valid ranges
        v_pert[:2 * N] = np.clip(v_pert[:2 * N], 1e-4, 1.0 - 1e-4)
        v_pert[2 * N:] = np.clip(v_pert[2 * N:], 1e-7, 0.5)
        
        try:
            res = minimize(
                objective_func, v_pert, method='SLSQP', bounds=bounds,
                constraints=constraints, options={'maxiter': 5000, 'ftol': 1e-13}
            )
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                c_vals = constraint_func(res.x)
                if np.min(c_vals) > -1e-7:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Extract final centers and radii
    centers = best_v[:2 * N].reshape(N, 2)
    radii = best_v[2 * N:].copy()
    
    # Phase 3: Strict Validation Repair (handles floating point drift)
    for _ in range(30):
        valid = True
        
        # Check boundaries
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            if x - r < -1e-12 or x + r > 1.0 + 1e-12 or y - r < -1e-12 or y + r > 1.0 + 1e-12:
                valid = False
                break
        if not valid:
            radii *= 0.9998
            continue
            
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            break
        radii *= 0.9998
        
    # Ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
