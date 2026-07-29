# sol_000062 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000034 (state 766fe0af) state=9b4cbc08 sum of radii=2.589106 correctness=1.0
# stdout(first 200): Running multi-start optimization on 26 initial configurations... Starting iterative refinement phase... Optimization complete. Best sum of radii: 2.589106
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_objective(vars_flat):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_flat[2 * N:])

def compute_constraints(vars_flat):
    """
    Computes all boundary and non-overlap constraints.
    Returns a 1D array where each element must be >= 0.
    """
    n = N
    centers = vars_flat[:2 * n].reshape(n, 2)
    radii = vars_flat[2 * n:]
    
    constraints = []
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    constraints.append(centers[:, 0] - radii)
    constraints.append(1.0 - centers[:, 0] - radii)
    constraints.append(centers[:, 1] - radii)
    constraints.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap: dist(i, j) >= r_i + r_j
    # Vectorized distance calculation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Upper triangle mask to avoid duplicates and self-comparison
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    constraints.append(dists[mask] - radii_sum[mask])
    
    return np.concatenate(constraints)

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    bounds = []
    for _ in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return bounds

def make_hex_init(row_counts, shift_pattern, seed, scale=0.95):
    """Generates a hexagonal lattice initialization with specified row structure."""
    rng = np.random.default_rng(seed)
    pts = []
    
    # Estimate spacing to fit N circles in unit square
    # Roughly 5-6 rows, 4-6 cols. Spacing ~0.18
    dx = 0.18
    dy = dx * np.sqrt(3.0) / 2.0
    
    y = 0.10
    for r_idx, count in enumerate(row_counts):
        shift = shift_pattern[r_idx] * (dx / 2.0)
        x = 0.10 + shift
        for _ in range(count):
            # Add small random noise to break symmetry
            px = x + rng.uniform(-0.005, 0.005)
            py = y + rng.uniform(-0.005, 0.005)
            pts.append([px, py])
            x += dx
        y += dy
        
    pts = np.array(pts[:N])
    
    # Scale and center to keep well inside boundaries initially
    pts = (pts - 0.5) * scale + 0.5
    
    # Compute safe initial radii
    radii = np.full(N, 0.05)
    for i in range(N):
        d_bound = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        d_neigh = min([np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]) 
                      for j in range(N) if i != j], default=2.0)
        radii[i] = min(d_bound, d_neigh * 0.5) * 0.9
        
    x0 = np.zeros(N * 3)
    for i in range(N):
        x0[3 * i] = pts[i, 0]
        x0[3 * i + 1] = pts[i, 1]
        x0[3 * i + 2] = radii[i]
    return x0

def make_random_init(seed):
    """Generates a random initial configuration."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.15, 0.85, (N, 2))
    radii = np.full(N, 0.04)
    
    x0 = np.zeros(N * 3)
    for i in range(N):
        x0[3 * i] = centers[i, 0]
        x0[3 * i + 1] = centers[i, 1]
        x0[3 * i + 2] = radii[i]
    return x0

def repair_solution(centers, radii):
    """Deterministically repairs centers and radii to guarantee strict feasibility."""
    n = centers.shape[0]
    radii = radii.copy()
    
    # Clamp centers to valid region based on radii
    for i in range(n):
        r = radii[i]
        centers[i, 0] = max(r, min(1.0 - r, centers[i, 0]))
        centers[i, 1] = max(r, min(1.0 - r, centers[i, 1]))
        
    # Iteratively shrink radii to resolve overlaps
    for _ in range(20):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if d < req - 1e-12:
                    shrink = (req - d + 1e-12) / 2.0
                    radii[i] -= shrink
                    radii[j] -= shrink
                    overlap_found = True
        if not overlap_found:
            break
            
    # Ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_val = -np.inf
    best_x = None
    
    # Phase 1: Multi-start optimization with diverse patterns
    # Row count configurations that sum to 26
    row_configs = [
        [5, 5, 5, 5, 5, 1], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4],
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [6, 4, 6, 5, 5],
        [5, 5, 5, 6, 5], [4, 5, 6, 5, 6]
    ]
    
    inits = []
    seed = 42
    for cfg in row_configs:
        shifts_even = [i % 2 for i in range(len(cfg))]
        shifts_odd = [(i + 1) % 2 for i in range(len(cfg))]
        inits.append(make_hex_init(cfg, shifts_even, seed))
        inits.append(make_hex_init(cfg, shifts_odd, seed))
        seed += 1
        
    # Add random inits
    for i in range(10):
        inits.append(make_random_init(seed + i))
        
    print(f"Running multi-start optimization on {len(inits)} initial configurations...")
    for x0 in inits:
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13})
            
            # Check feasibility with tolerance
            c_vals = compute_constraints(res.x)
            if np.all(c_vals >= -1e-7):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
            
    if best_x is None:
        best_x = inits[0]
        
    # Phase 2: Iterative Perturb & Optimize (Basin Hopping style)
    print("Starting iterative refinement phase...")
    curr_x = best_x.copy()
    for step in range(30):
        # Perturb centers and radii slightly
        pert_x = curr_x.copy()
        pert_x[:2 * N] += np.random.normal(0, 0.002, 2 * N)
        pert_x[2 * N:] *= np.random.uniform(0.995, 1.005, N)
        
        try:
            res = minimize(compute_objective, pert_x, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-13})
            
            c_vals = compute_constraints(res.x)
            if np.all(c_vals >= -1e-7):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
                    curr_x = best_x.copy()
        except Exception:
            pass
            
    print(f"Optimization complete. Best sum of radii: {best_val:.6f}")
    
    # Extract and repair final solution
    centers = best_x[:2 * N].reshape(N, 2)
    radii = best_x[2 * N:].copy()
    
    centers, radii = repair_solution(centers, radii)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
