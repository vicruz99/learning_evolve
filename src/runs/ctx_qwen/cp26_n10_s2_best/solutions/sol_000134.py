# sol_000134 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000107 (state 1a0a7ebc) state=88e51fb3 sum of radii=2.626572 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
NUM_PAIRS = len(PAIR_I)

def objective(v):
    """Objective: Minimize negative sum of radii (maximize sum)."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap (squared for stability)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + NUM_PAIRS)
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return c

def get_feasible_r(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    wall_dists = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Initialize at 92% of theoretical max to guarantee strict feasibility
    return 0.92 * np.minimum(min_dists / 2.0, wall_dists)

def make_init(seed, type_='hex'):
    """Generate a diverse initial configuration."""
    np.random.seed(seed)
    if type_ == 'hex':
        pts = []
        r0 = 0.10 + np.random.uniform(-0.02, 0.02)
        for i in range(-6, 10):
            for j in range(-6, 10):
                x = i * r0 + (j % 2) * r0 * 0.5
                y = j * r0 * np.sqrt(3) * 0.5
                pts.append([x, y])
        pts = np.array(pts)
        
        ang = np.random.uniform(-0.3, 0.3)
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s], [s, c]])
        pts = (pts - pts.mean(axis=0)) @ rot.T
        pts += [0.5 + np.random.uniform(-0.05, 0.05), 0.5 + np.random.uniform(-0.05, 0.05)]
        
        mask = (pts[:, 0] > 0.05) & (pts[:, 0] < 0.95) & (pts[:, 1] > 0.05) & (pts[:, 1] < 0.95)
        pts = pts[mask]
        
        if len(pts) < N:
            pts = np.random.uniform(0.1, 0.9, (N, 2))
        else:
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
    elif type_ == 'grid':
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.1 + i*0.15 + np.random.uniform(-0.01, 0.01), 
                            0.1 + j*0.18 + np.random.uniform(-0.01, 0.01)])
        pts = np.array(pts[:N])
    else:
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        
    return pts

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Multi-start exploration
    inits = []
    for s in range(25):
        inits.append(make_init(s, 'hex'))
    for s in range(10):
        inits.append(make_init(s, 'grid'))
    for s in range(10):
        inits.append(make_init(s, 'rand'))
        
    for centers in inits:
        r_init = get_feasible_r(centers)
        v0 = np.concatenate([centers[:, 0], centers[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            
            if -res.fun > best_sum:
                if np.min(constraints(res.x)) >= -1e-7:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation refinement to escape local minima
    if best_v is not None:
        current_v = best_v.copy()
        for step in range(40):
            np.random.seed(step * 31 + 7)
            v_pert = current_v.copy()
            v_pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.02, 0.98)
            
            # Shrink radii to guarantee feasibility after perturbation
            v_pert[2*N:] *= 0.95
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                
                if -res.fun > best_sum:
                    if np.min(constraints(res.x)) >= -1e-7:
                        best_sum = -res.fun
                        best_v = res.x.copy()
                        current_v = best_v.copy()
            except Exception:
                continue
                
    # Phase 3: Radius growth & center relaxation
    if best_v is not None:
        current_v = best_v.copy()
        for _ in range(15):
            current_v[2*N:] *= 1.003
            current_v[:2*N] += np.random.uniform(-0.002, 0.002, 2*N)
            current_v[:2*N] = np.clip(current_v[:2*N], 0.01, 0.99)
            
            try:
                res = minimize(objective, current_v, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
                
                if -res.fun > best_sum:
                    if np.min(constraints(res.x)) >= -1e-7:
                        best_sum = -res.fun
                        best_v = res.x.copy()
                        current_v = best_v.copy()
            except Exception:
                continue

    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 4: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with minimal safety margin
    for _ in range(5):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
