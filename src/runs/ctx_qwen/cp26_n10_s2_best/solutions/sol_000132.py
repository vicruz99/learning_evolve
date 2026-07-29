# sol_000132 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000107 (state 1a0a7ebc) state=1b9f9dae sum of radii=2.618481 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)
BOUNDS = [(0.0, 1.0)] * (2 * N) + [(1e-5, 0.25)] * N

def objective(v):
    """Objective: Minimize negative sum of radii (maximize sum)."""
    return -np.sum(v[2 * N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap (squared)."""
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    
    c = np.empty(4 * N + NUM_PAIRS)
    c[:N] = x - r
    c[N:2 * N] = 1.0 - x - r
    c[2 * N:3 * N] = y - r
    c[3 * N:4 * N] = 1.0 - y - r
    
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4 * N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def compute_feasible_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    wall_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # 90% of max theoretical radius guarantees strict feasibility
    r = 0.90 * np.minimum(min_dists / 2.0, wall_dists)
    return np.clip(r, 0.005, 0.25)

def force_relax(centers, iters=250):
    """Quick repulsive force simulation to spread points into a feasible layout."""
    pts = centers.copy()
    dt = 0.004
    for _ in range(iters):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                if d < 0.22 and d > 1e-5:
                    f = (0.22 - d) / d * 0.6
                    diff = pts[i] - pts[j]
                    forces[i] += diff * f
                    forces[j] -= diff * f
        pts += forces * dt
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_single_opt(v0):
    """Run SLSQP optimization and return result safely."""
    try:
        res = minimize(
            objective, v0, method='SLSQP', bounds=BOUNDS,
            constraints={'type': 'ineq', 'fun': constraints},
            options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False}
        )
        return res
    except Exception:
        return None

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    best_sum = -1.0
    best_v = None
    
    inits = []
    
    # 1. Hexagonal lattices with rotation and force relaxation
    for seed in range(12):
        np.random.seed(seed)
        angle = np.random.uniform(-0.25, 0.25)
        pts = []
        r0 = 0.105
        for i in range(-7, 9):
            for j in range(-7, 9):
                x = i * r0 + (j % 2) * r0 * 0.5
                y = j * r0 * np.sqrt(3) * 0.5
                pts.append([x, y])
        pts = np.array(pts)
        
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
        pts -= pts.mean(axis=0)
        pts += [0.5, 0.5]
        
        mask = (pts[:, 0] > 0.05) & (pts[:, 0] < 0.95) & \
               (pts[:, 1] > 0.05) & (pts[:, 1] < 0.95)
        valid = pts[mask]
        if len(valid) >= N:
            idx = np.random.choice(len(valid), N, replace=False)
            pts_sel = valid[idx]
            pts_sel = force_relax(pts_sel, iters=200)
            inits.append(pts_sel)
            
    # 2. Random dense scatter with relaxation
    for seed in range(12):
        np.random.seed(seed + 1000)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        pts = force_relax(pts, iters=300)
        inits.append(pts)
        
    # Phase 1: Multi-start optimization
    for centers in inits:
        r_init = compute_feasible_radii(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        res = run_single_opt(v0)
        if res is not None and -res.fun > best_sum:
            c_val = constraints(res.x)
            if np.min(c_val) >= -1e-7:
                best_sum = -res.fun
                best_v = res.x.copy()
                
    # Phase 2: Adaptive Perturbation & Refinement
    # Escapes local minima by shrinking radii, jittering centers, and re-optimizing
    if best_v is not None:
        for step in range(50):
            np.random.seed(step + 5000)
            v_pert = best_v.copy()
            
            # Gradually decrease shrinkage to focus search as we converge
            shrink = 0.96 - step * 0.003
            v_pert[2 * N:] *= max(0.90, shrink)
            
            # Perturb centers
            v_pert[:2 * N] += np.random.uniform(-0.003, 0.003, 2 * N)
            v_pert[:2 * N] = np.clip(v_pert[:2 * N], 0.02, 0.98)
            
            # Recompute strictly feasible radii for the new center positions
            c_pert = v_pert[:2 * N].reshape(N, 2)
            v_pert[2 * N:] = compute_feasible_radii(c_pert) * 0.95
            
            res = run_single_opt(v_pert)
            if res is not None and -res.fun > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = -res.fun
                    best_v = res.x.copy()
                    
    # Fallback initialization if optimization completely fails (extremely unlikely)
    if best_v is None:
        centers_fb = np.random.uniform(0.2, 0.8, (N, 2))
        centers_fb = force_relax(centers_fb, iters=400)
        r_fb = compute_feasible_radii(centers_fb)
        best_v = np.concatenate([centers_fb[:, 0], centers_fb[:, 1], r_fb])
        best_sum = np.sum(r_fb)
        
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2 * N]))
    radii = best_v[2 * N:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with safety margin
    for _ in range(15):
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
            
    return centers, radii, float(np.sum(radii))
