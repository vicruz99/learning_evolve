# sol_000183 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000167 (state d81766f0) state=c055733b sum of radii=2.630730 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and squared non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    r_sum = r[PAIR_I] + r[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - r_sum**2
    
    return c

def get_feasible_radii(centers, scale=0.85):
    """Compute strictly feasible initial radii based on local geometry."""
    # Distance to boundaries
    r = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Distance to other centers
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    r = np.minimum(r, min_dists / 2.0)
    
    return np.clip(r * scale, 1e-6, 0.5)

def generate_inits():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal lattice variations
    patterns = [[5,6,5,6,4], [6,5,6,5,4], [5,5,5,5,6], [4,6,6,6,4], [6,6,5,5,4], [7,5,5,5,4]]
    for pat in patterns:
        if sum(pat) < N: 
            continue
        r0 = 0.095
        pts = []
        y = r0
        row_idx = 0
        for count in pat:
            x_start = r0 if row_idx % 2 == 0 else 2.0 * r0
            for k in range(count):
                if len(pts) >= N: 
                    break
                pts.append([x_start + k * 2.0 * r0, y])
            y += r0 * np.sqrt(3.0)
            row_idx += 1
        base = np.array(pts[:N])
        
        for ang in np.linspace(-0.12, 0.12, 7):
            for sx in np.linspace(-0.025, 0.025, 4):
                for sy in np.linspace(-0.025, 0.025, 4):
                    p = base.copy()
                    if abs(ang) > 1e-6:
                        c_val, s_val = np.cos(ang), np.sin(ang)
                        p = (p - 0.5) @ np.array([[c_val, -s_val], [s_val, c_val]]) + 0.5
                    p[:, 0] += sx
                    p[:, 1] += sy
                    if np.all((p[:,0] >= 0.02) & (p[:,0] <= 0.98) & (p[:,1] >= 0.02) & (p[:,1] <= 0.98)):
                        r_init = get_feasible_radii(p, 0.82)
                        inits.append(np.concatenate([p[:,0], p[:,1], r_init]))

    # 2. Force-relaxed random configurations
    for seed in range(25):
        np.random.seed(seed)
        pts = np.random.uniform(0.12, 0.88, (N, 2))
        # Repulsion relaxation
        for _ in range(120):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    d = np.hypot(diff[0], diff[1])
                    if d < 0.22 and d > 1e-5:
                        f = (0.22 - d) / d
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.04
            pts = np.clip(pts, 0.05, 0.95)
        r_init = get_feasible_radii(pts, 0.80)
        inits.append(np.concatenate([pts[:,0], pts[:,1], r_init]))
        
    return inits

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    inits = generate_inits()
    
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # Phase 1: Multi-start optimization
    for v0 in inits:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 40000, 'ftol': 1e-14, 'disp': False})
            s = -res.fun
            if s > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        # Fallback
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        r_fb = get_feasible_radii(pts, 0.7)
        best_v = np.concatenate([pts[:,0], pts[:,1], r_fb])
        best_sum = -np.sum(r_fb)
        
    # Phase 2: Iterative perturbation & refinement to escape local minima
    current_v = best_v.copy()
    for step in range(50):
        np.random.seed(step + 6000)
        v_pert = current_v.copy()
        
        # Decaying noise schedule
        noise_scale = 0.004 * (1.0 - step / 50.0)
        v_pert[:2*N] += np.random.normal(0, noise_scale, 2*N)
        v_pert[:2*N] = np.clip(v_pert[:2*N], 0.02, 0.98)
        
        # Shrink radii to create slack for rearrangement
        v_pert[2*N:] *= 0.960
        
        # Recompute feasible radii to guarantee strict feasibility
        c_pts = v_pert[:2*N].reshape(N, 2)
        v_pert[2*N:] = get_feasible_radii(c_pts, 0.85)
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 30000, 'ftol': 1e-14, 'disp': False})
            s = -res.fun
            if s > best_sum:
                cv = constraints(res.x)
                if np.min(cv) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            pass
            
    # Extract results
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:,0], 1.0-centers[:,0]))
    radii = np.minimum(radii, np.minimum(centers[:,1], 1.0-centers[:,1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with minimal safety margin
    # Validator allows dist >= r1 + r2 - 1e-12. We use a tiny buffer.
    for _ in range(30):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d + 1e-13:
                    shrink = (radii[i] + radii[j] - d - 1e-13) / 2.0
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
