# sol_000118 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000083 (state c6ee3a07) state=05734ab4 sum of radii=2.626572 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pair indices for efficient constraint evaluation
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + len(PAIR_I))
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    return c

def make_feasible(centers, radii):
    """Adjusts radii to guarantee strict feasibility for a given center layout."""
    r = radii.copy()
    # Boundary constraints
    r = np.minimum(r, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    r = np.minimum(r, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Pairwise constraints (iterative shrinkage)
    for _ in range(5):
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if r[i] + r[j] > d:
                    shrink = (r[i] + r[j] - d) / 2.0 + 1e-9
                    r[i] = max(0.0, r[i] - shrink)
                    r[j] = max(0.0, r[j] - shrink)
    return r

def generate_initialization(seed, layout_type):
    """Generates a strictly feasible initial configuration."""
    np.random.seed(seed)
    centers = np.zeros((N, 2))
    
    if layout_type == 'hex':
        r0 = 0.09 + 0.015 * np.random.rand()
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 5:
            x_start = r0 + (row % 2) * r0
            x = x_start
            while x <= 1.0 - r0:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        centers = np.array(pts[:N])
    elif layout_type == 'grid':
        pts = []
        for i in range(6):
            for j in range(5):
                if len(pts) < N:
                    pts.append([0.1 + i * 0.16 + 0.01 * np.random.rand(), 
                                0.1 + j * 0.18 + 0.01 * np.random.rand()])
        centers = np.array(pts)
    else: # random
        centers = np.random.uniform(0.15, 0.85, (N, 2))
        
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.08, 0.92)
    
    # Start with moderate radii and force feasibility
    r_init = np.full(N, 0.05)
    radii = make_feasible(centers, r_init)
    
    return np.concatenate([centers[:, 0], centers[:, 1], radii])

def run_packing():
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Diverse multi-start optimization
    for layout in ['hex', 'grid', 'rand']:
        for seed in range(12):
            v0 = generate_initialization(seed, layout)
            try:
                res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 12000, 'ftol': 1e-12, 'disp': False})
                curr_sum = -res.fun
                # Check feasibility tolerance
                if np.all(constraints(res.x) >= -1e-6) and curr_sum > best_sum:
                    best_sum = curr_sum
                    best_v = res.x.copy()
            except Exception:
                continue
                
    if best_v is None:
        # Fallback safe config
        centers = np.random.rand(N, 2) * 0.6 + 0.2
        radii = make_feasible(centers, np.full(N, 0.03))
        return centers, radii, float(np.sum(radii))
        
    # Phase 2: Perturbation search to escape local minima
    for step in range(12):
        v_pert = best_v.copy()
        # Decay perturbation magnitude over steps
        scale = 0.004 * (1.0 - step / 15.0)
        noise = np.random.uniform(-scale, scale, 3 * N)
        noise[2*N:] *= 0.3 # Perturb centers more than radii
        v_pert += noise
        v_pert = np.clip(v_pert, 0.0, 1.0)
        v_pert[2*N:] = np.clip(v_pert[2*N:], 0.0, 0.5)
        
        try:
            res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            if np.all(constraints(res.x) >= -1e-6) and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
        except Exception:
            continue
            
    # Phase 3: Controlled inflation to push boundaries
    v_inf = best_v.copy()
    for _ in range(6):
        v_inf[2*N:] *= 1.0025
        try:
            res = minimize(objective, v_inf, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            curr_sum = -res.fun
            if np.all(constraints(res.x) >= -1e-6) and curr_sum > best_sum:
                best_sum = curr_sum
                best_v = res.x.copy()
                v_inf = best_v.copy()
        except Exception:
            break
            
    # Extract and strictly enforce validity
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Post-processing to guarantee validator compliance
    for _ in range(15):
        changed = False
        # Boundary clamp
        for i in range(N):
            mr = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            if radii[i] > mr:
                radii[i] = mr
                changed = True
                
        # Overlap resolution
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if radii[i] + radii[j] > d:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
