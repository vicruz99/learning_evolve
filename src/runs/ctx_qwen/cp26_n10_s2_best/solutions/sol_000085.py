# sol_000085 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000049 (state 0aad4082) state=d65ecbba sum of radii=2.617761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pairwise indices for efficient constraint evaluation
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Inequality constraints: boundaries and non-overlap."""
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
    dr = r[PAIR_I] + r[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - dr**2
    
    return c

def generate_lattice_start(rotation=0.0, scale_factor=1.0):
    """Generates a rotated hexagonal lattice initialization."""
    # 5 rows pattern: 6, 5, 6, 5, 4 circles
    counts = [6, 5, 6, 5, 4]
    pts = []
    y = 0.0
    for idx, cnt in enumerate(counts):
        x_start = 0.5 if idx % 2 == 1 else 0.0
        for i in range(cnt):
            pts.append([x_start + i, y])
        y += np.sqrt(3)/2
    pts = np.array(pts)
    
    # Center and scale to fit roughly in unit square
    pts -= pts.mean(axis=0)
    max_extent = np.max(np.abs(pts), axis=0).max()
    pts /= max_extent * 0.75 * scale_factor
    
    # Apply rotation
    if rotation != 0:
        c, s = np.cos(rotation), np.sin(rotation)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
        
    # Shift to [0, 1] domain with margins
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)
    pts = pts * 0.85 + 0.075
    
    # Small initial radii to guarantee feasibility
    r0 = 0.015
    return np.concatenate([pts[:,0], pts[:,1], np.full(N, r0)])

def run_packing():
    bounds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_val = -np.inf
    
    # Phase 1: Generate diverse lattice starts
    starts = []
    for ang in np.linspace(0, 0.3, 9):
        for sc in [0.9, 1.0, 1.1]:
            starts.append(generate_lattice_start(ang, sc))
            
    # Phase 2: Add random dense starts
    np.random.seed(42)
    for seed in range(10):
        pts = np.random.uniform(0.1, 0.9, (N, 2))
        v0 = np.concatenate([pts[:,0], pts[:,1], np.full(N, 0.02)])
        starts.append(v0)
        
    # Run optimization from each start
    for v0 in starts:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13})
            if -res.fun > best_val:
                cv = constraints(res.x)
                if np.all(cv >= -1e-5):
                    best_val = -res.fun
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 3: Local perturbation & refinement to escape local minima
    if best_v is not None:
        for _ in range(15):
            vp = best_v.copy()
            vp[:2*N] += np.random.uniform(-0.008, 0.008, 2*N)
            vp[:2*N] = np.clip(vp[:2*N], 0.02, 0.98)
            vp[2*N:] *= 0.94  # Shrink radii to ensure feasibility after perturbation
            try:
                res = minimize(objective, vp, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13})
                if -res.fun > best_val and np.all(constraints(res.x) >= -1e-5):
                    best_val = -res.fun
                    best_v = res.x.copy()
            except Exception:
                pass
                
    # Fallback if optimization completely fails
    if best_v is None:
        best_v = generate_lattice_start(0.0)
        
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:]
    
    centers = np.column_stack((cx, cy))
    radii = cr.copy()
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints
    radii = np.minimum(radii, np.minimum(centers[:,0], 1.0 - centers[:,0]))
    radii = np.minimum(radii, np.minimum(centers[:,1], 1.0 - centers[:,1]))
    
    # 2. Enforce non-overlap constraints iteratively
    for _ in range(30):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                if radii[i] + radii[j] > d - 1e-10:
                    shrink = (radii[i] + radii[j] - d)/2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
