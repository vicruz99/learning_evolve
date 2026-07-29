# sol_000075 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000054 (state 91c332d2) state=4372890b sum of radii=2.620237 correctness=1.0
# stdout(first 200): Running 30 initial optimizations... Optimization complete. Final sum of radii: 2.620237
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_constraints(v):
    """
    Computes all boundary and non-overlap constraints.
    Returns a 1D array where each element must be >= 0 for feasibility.
    """
    centers = v[:2 * N].reshape(N, 2)
    radii = v[2 * N:]
    
    con = []
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    con.append(centers[:, 0] - radii)
    con.append(1.0 - centers[:, 0] - radii)
    con.append(centers[:, 1] - radii)
    con.append(1.0 - centers[:, 1] - radii)
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    idx = np.triu_indices(N, k=1)
    dx = centers[idx[0], 0] - centers[idx[1], 0]
    dy = centers[idx[0], 1] - centers[idx[1], 1]
    dr = radii[idx[0]] + radii[idx[1]]
    con.append(dx**2 + dy**2 - dr**2)
    
    return np.concatenate(con)

def objective(v):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2 * N:])

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    return [(0.0, 1.0)] * (2 * N) + [(1e-7, 0.5)] * N

def hex_init(r, seed=0):
    """Generates a hexagonal lattice initialization with specified radius estimate."""
    rng = np.random.default_rng(seed)
    pts = []
    y = r
    row = 0
    while len(pts) < N:
        x = r + (row % 2) * r
        while x + r <= 1.0 and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r
        y += r * np.sqrt(3.0)
        row += 1
    pts = np.array(pts[:N])
    # Add small noise to break symmetry and avoid degenerate flat landscapes
    pts += rng.normal(0, 0.002, pts.shape)
    return np.clip(pts, 0.05, 0.95)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Packs 26 circles in a unit square to maximize the sum of radii."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_v = None
    best_sum = -np.inf
    
    # Phase 1: Multi-start global search with diverse patterns
    starts = []
    
    # Hexagonal lattices with varying densities
    for r in [0.090, 0.095, 0.100, 0.105, 0.108]:
        for seed in range(4):
            c = hex_init(r, seed)
            rad = np.full(N, r * 0.98)  # Slightly shrink to ensure initial feasibility
            starts.append(np.concatenate([c.flatten(), rad]))
            
    # Random dense configurations
    for seed in range(10):
        rng = np.random.default_rng(seed)
        c = rng.uniform(0.15, 0.85, (N, 2))
        rad = np.full(N, 0.08)
        starts.append(np.concatenate([c.flatten(), rad]))
        
    print(f"Running {len(starts)} initial optimizations...")
    for v0 in starts:
        try:
            res = minimize(
                objective, v0, method='SLSQP', bounds=bounds,
                constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False}
            )
            curr_sum = -res.fun
            # Check feasibility with tolerance
            c_vals = compute_constraints(res.x)
            if np.min(c_vals) > -1e-7:
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        best_v = starts[0]
        
    # Phase 2: Perturb & Scale optimization to escape local minima
    # This forces the optimizer to search for configurations supporting larger radii
    rng = np.random.default_rng(42)
    for step in range(30):
        noise_scale = 0.002 * (0.88 ** step)
        v_curr = best_v.copy()
        
        # Perturb centers
        v_curr[:2 * N] += rng.normal(0, noise_scale, 2 * N)
        v_curr[:2 * N] = np.clip(v_curr[:2 * N], 0.0, 1.0)
        
        # Slightly inflate radii to push towards tighter packing
        v_curr[2 * N:] *= (1.0 + 0.001 * np.exp(-step/10.0))
        v_curr[2 * N:] = np.clip(v_curr[2 * N:], 1e-7, 0.5)
        
        try:
            res = minimize(
                objective, v_curr, method='SLSQP', bounds=bounds,
                constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False}
            )
            curr_sum = -res.fun
            c_vals = compute_constraints(res.x)
            if np.min(c_vals) > -1e-7:
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Extract results
    centers = best_v[:2 * N].reshape(N, 2)
    radii = best_v[2 * N:].copy()
    
    # Phase 3: Strict validation repair (handles floating point drift)
    for _ in range(50):
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
        
    final_sum = float(np.sum(radii))
    print(f"Optimization complete. Final sum of radii: {final_sum:.6f}")
    return centers, radii, final_sum
