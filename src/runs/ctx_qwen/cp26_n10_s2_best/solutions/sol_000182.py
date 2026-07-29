# sol_000182 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000167 (state d81766f0) state=9525cb66 sum of radii=2.626740 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and squared non-overlap distances."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    c = np.empty(4*N + len(PAIR_I))
    # Boundary constraints
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    # Pairwise non-overlap constraints (squared for stability)
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    return c

def compute_feasible_radii(centers, shrink=0.85):
    """Compute strictly feasible initial radii based on local geometry."""
    # Distance to walls
    r = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    # Distance to other centers
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    r = np.minimum(r, np.min(dists, axis=1) / 2.0)
    # Scale down to guarantee strict feasibility
    return np.clip(r * shrink, 1e-4, 0.3)

def generate_hex_config(pattern, r0, angle=0.0, dx=0.0, dy=0.0):
    """Generate a hexagonal lattice configuration based on row pattern."""
    pts = []
    y = r0 + dy
    row_idx = 0
    for c in pattern:
        x_start = r0 + dx + (row_idx % 2) * r0
        for k in range(c):
            if len(pts) >= N: break
            x = x_start + k * 2.0 * r0
            pts.append([x, y])
        y += np.sqrt(3.0) * r0
        row_idx += 1
    pts = np.array(pts[:N])
    if abs(angle) > 1e-6:
        c_val, s_val = np.cos(angle), np.sin(angle)
        pts = (pts - 0.5) @ np.array([[c_val, -s_val], [s_val, c_val]]) + 0.5
    return pts

def force_relax(centers, iters=250):
    """Relax configuration using repulsive forces to find dense local packing."""
    pts = centers.copy()
    target_dist = 0.25
    lr = 0.05
    for _ in range(iters):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        mask = dists < target_dist
        f_mag = np.zeros_like(dists)
        f_mag[mask] = (target_dist - dists[mask]) / dists[mask]
        f_vec = diff * f_mag[:, :, None]
        forces = np.sum(f_vec, axis=1)
        pts += forces * lr
        pts = np.clip(pts, 0.02, 0.98)
        lr *= 0.998
    return pts

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-5, 0.3)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    inits = []
    
    # 1. Hexagonal patterns with various row distributions and parameters
    patterns = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [6, 6, 5, 5, 4],
        [5, 5, 6, 6, 4],
        [7, 5, 5, 5, 4],
        [5, 5, 5, 5, 6]
    ]
    for pat in patterns:
        if sum(pat) < N: continue
        for r0 in [0.09, 0.095, 0.10, 0.105]:
            for ang in np.linspace(-0.15, 0.15, 5):
                for sx in [-0.02, 0.0, 0.02]:
                    for sy in [-0.02, 0.0, 0.02]:
                        p = generate_hex_config(pat, r0, ang, sx, sy)
                        if len(p) < N: continue
                        if np.all((p[:,0] > 0.01) & (p[:,0] < 0.99) & (p[:,1] > 0.01) & (p[:,1] < 0.99)):
                            inits.append(p.copy())

    # 2. Force-relaxed random configurations
    for seed in range(15):
        np.random.seed(seed + 1000)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        pts = force_relax(pts, iters=200)
        inits.append(pts)
        
    # 3. Staggered grid variations
    for s in np.linspace(0.0, 0.03, 4):
        pts = []
        for i in range(6):
            for j in range(5):
                if len(pts) >= N: break
                pts.append([0.08 + i*0.16 + s, 0.08 + j*0.20 + s])
        if len(pts) >= N:
            inits.append(np.array(pts[:N]))

    # Phase 1: Multi-start optimization
    for centers in inits:
        r_init = compute_feasible_radii(centers, shrink=0.80)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 60000, 'ftol': 1e-14, 'disp': False})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-5:
                    best_sum = curr_sum
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Fallback initialization if all fail
    if best_v is None:
        c_f = np.random.uniform(0.1, 0.9, (N, 2))
        r_f = compute_feasible_radii(c_f)
        best_v = np.concatenate([c_f[:,0], c_f[:,1], r_f])
        best_sum = -np.sum(r_f)

    # Phase 2: Iterative perturbation & refinement to escape local minima
    current_v = best_v.copy()
    for step in range(60):
        np.random.seed(step + 500)
        v_p = current_v.copy()
        
        # Gradually decreasing noise scale
        noise = 0.004 * (1.0 - step / 60.0)
        v_p[:2*N] += np.random.normal(0, noise, 2*N)
        v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
        
        # Shrink radii to create slack and guarantee restart feasibility
        centers_p = v_p[:2*N].reshape(N, 2)
        v_p[2*N:] = compute_feasible_radii(centers_p, shrink=0.92)
        
        try:
            res = minimize(objective, v_p, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 50000, 'ftol': 1e-14, 'disp': False})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-5:
                    best_sum = curr_sum
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            pass

    # Extract results
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # 2. Enforce non-overlap constraints iteratively with minimal safety margin
    # Validator allows: dist >= r1 + r2 - 1e-12. We use 1e-13 buffer to be safe.
    for _ in range(25):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d + 1e-13:
                    shrink = (radii[i] + radii[j] - d - 1e-13) / 2.0
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
