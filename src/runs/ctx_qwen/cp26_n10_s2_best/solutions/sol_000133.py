# sol_000133 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000107 (state 1a0a7ebc) state=d6838e2f sum of radii=2.628083 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii (equivalent to maximizing sum)."""
    return -np.sum(v[2*N:])

def constraints(v):
    """
    Compute inequality constraints: boundaries and non-overlap.
    Uses squared distances for smoother gradients and numerical stability.
    All returned values must be >= 0.
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + len(PAIR_I))
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def compute_feasible_radii(centers):
    """
    Computes strictly feasible radii for a given set of centers.
    Takes 90% of the theoretical maximum to guarantee a feasible starting point.
    """
    # Distance to boundaries
    r = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Distance to nearest neighbor
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r = np.minimum(r, np.min(dists, axis=1) / 2.0)
    
    return np.clip(r * 0.90, 1e-4, 0.45)

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # --- Phase 1: Diverse Multi-Start Exploration ---
    initial_centers = []
    
    # 1. Hexagonal lattices with various densities and shifts
    for seed in range(12):
        np.random.seed(seed)
        r0 = 0.095 + np.random.uniform(-0.005, 0.010)
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 8:
            x_start = r0 if row % 2 == 0 else 2.0 * r0
            x = x_start
            while x <= 1.0 - r0 and len(pts) < N + 8:
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3.0)
            row += 1
        pts = np.array(pts[:N])
        pts += np.random.uniform(-0.015, 0.015, pts.shape)
        initial_centers.append(np.clip(pts, 0.05, 0.95))
        
    # 2. Staggered grids
    for seed in range(8):
        np.random.seed(seed + 100)
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.08 + i*0.165 + np.random.uniform(-0.01, 0.01),
                            0.08 + j*0.195 + np.random.uniform(-0.01, 0.01)])
        initial_centers.append(np.array(pts[:N]))
        
    # 3. Uniform random scatter
    for seed in range(10):
        np.random.seed(seed + 200)
        initial_centers.append(np.random.uniform(0.12, 0.88, (N, 2)))
        
    # Optimize from each start
    for centers in initial_centers:
        v0 = np.concatenate([centers[:, 0], centers[:, 1], compute_feasible_radii(centers)])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(constraints(res.x)) >= -1e-7:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Fallback initialization if all fails
    if best_v is None:
        v0 = np.zeros(3*N)
        v0[:2*N] = np.random.uniform(0.2, 0.8, 2*N)
        v0[2*N:] = 0.03
        best_v = v0
        best_sum = -np.sum(v0[2*N:])
        
    # --- Phase 2: Iterative Shrink-Perturb-Optimize Refinement ---
    current_v = best_v.copy()
    for step in range(60):
        np.random.seed(step + 1000)
        v_p = current_v.copy()
        
        # Scheduled shrinkage: start larger to break constraint locks, decrease over time
        shrink_factor = 0.92 + 0.07 * (step / 60.0)
        v_p[2*N:] *= shrink_factor
        
        # Perturb centers: larger early on, fine-tune later
        noise_scale = 0.009 * (1.0 - step / 60.0)
        v_p[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
        v_p[:2*N] = np.clip(v_p[:2*N], 0.03, 0.97)
        
        # Recompute strictly feasible radii for the perturbed centers
        v_p[2*N:] = compute_feasible_radii(v_p[:2*N].reshape(N, 2))
        
        try:
            res = minimize(objective, v_p, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            
            # Accept only if strictly feasible and improves sum
            if res.success and np.min(constraints(res.x)) >= -1e-7:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            continue
            
    # --- Phase 3: Strict Post-Processing for Validator Compliance ---
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(25):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    final_sum = float(np.sum(radii))
    return centers, radii, final_sum
