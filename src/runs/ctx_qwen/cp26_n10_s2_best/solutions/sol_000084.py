# sol_000084 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000049 (state 0aad4082) state=2a8c4dca sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
MASK = np.triu(np.ones((N, N), dtype=bool), k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + N*(N-1)//2)
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap: dist >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    rs = r[:, None] + r[None, :]
    c[4*N:] = (dist - rs)[MASK]
    return c

def get_initial_config(seed, r_base, rot=0.0):
    """Generates a feasible initial configuration from a perturbed hexagonal lattice."""
    np.random.seed(seed)
    pts = []
    y = r_base
    row = 0
    while len(pts) < N + 10:
        x_start = r_base + (row % 2) * r_base
        x = x_start
        while x <= 1.0 - r_base:
            pts.append([x, y])
            x += 2.0 * r_base
        y += r_base * np.sqrt(3)
        row += 1
        
    pts = np.array(pts[:N])
    
    # Apply rotation if specified
    if rot != 0.0:
        cx, cy = 0.5, 0.5
        pts -= [cx, cy]
        c, s = np.cos(rot), np.sin(rot)
        pts = pts @ np.array([[c, -s], [s, c]])
        pts += [cx, cy]
        
    # Add controlled jitter to break symmetry
    pts += np.random.uniform(-0.01, 0.01, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    
    # Compute strictly feasible initial radii based on geometry
    r = np.full(N, 0.5)
    for i in range(N):
        mr = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        for j in range(N):
            if i == j: continue
            d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
            if d/2.0 < mr: mr = d/2.0
        r[i] = mr
        
    # Scale down to guarantee initial feasibility and allow optimizer expansion
    r *= 0.70 
    
    v = np.zeros(3*N)
    v[:N] = pts[:, 0]
    v[N:2*N] = pts[:, 1]
    v[2*N:] = r
    return v

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_val = -np.inf
    
    # Phase 1: Multi-start optimization with diverse lattice configurations
    starts = []
    for r_b in [0.075, 0.085, 0.095, 0.105, 0.115]:
        for rot in [0.0, 0.2, 0.4]:
            for seed in range(5):
                starts.append((r_b, rot, seed))
                
    for r_b, rot, seed in starts:
        v0 = get_initial_config(seed, r_b, rot)
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_val:
                if np.min(constraints(res.x)) >= -1e-8:
                    best_val = -res.fun
                    best_v = res.x.copy()
        except:
            pass
            
    if best_v is None:
        best_v = get_initial_config(0, 0.09)
        
    # Phase 2: Local refinement to escape shallow local minima
    for step in range(12):
        v0 = best_v.copy()
        # Perturb centers slightly
        v0[:2*N] += np.random.uniform(-0.006, 0.006, 2*N)
        v0[:2*N] = np.clip(v0[:2*N], 0.02, 0.98)
        # Shrink radii to ensure feasibility after perturbation
        v0[2*N:] *= 0.96
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_val:
                if np.min(constraints(res.x)) >= -1e-8:
                    best_val = -res.fun
                    best_v = res.x.copy()
        except:
            pass
            
    # Phase 3: Iterative radius expansion to squeeze out maximum possible sum
    for _ in range(25):
        v0 = best_v.copy()
        v0[2*N:] *= 1.001  # Deliberately expand radii slightly
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_val:
                if np.min(constraints(res.x)) >= -1e-8:
                    best_val = -res.fun
                    best_v = res.x.copy()
            else:
                break  # Convergence reached
        except:
            break
            
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Strict post-processing to guarantee validator tolerance compliance
    eps = 1e-9
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        radii[i] = min(radii[i], centers[i,0]-eps, 1.0-centers[i,0]-eps, 
                       centers[i,1]-eps, 1.0-centers[i,1]-eps)
    # 2. Enforce non-overlap constraints strictly
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            if d < radii[i] + radii[j] - eps:
                shrink = (radii[i] + radii[j] - d) / 2.0 + eps
                radii[i] = max(0.0, radii[i] - shrink)
                radii[j] = max(0.0, radii[j] - shrink)
                
    return centers, radii, float(np.sum(radii))
