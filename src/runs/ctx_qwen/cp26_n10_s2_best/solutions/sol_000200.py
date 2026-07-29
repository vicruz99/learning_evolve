# sol_000200 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000128 (state 9970da69) state=03360110 sum of radii=2.634292 correctness=1.0
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
    """Inequality constraints: boundaries and squared non-overlap distances."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    c = np.empty(4*N + len(PAIR_I))
    # Boundary constraints
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap constraints
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c[4*N:] = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    return c

def get_feasible_radii(centers, shrink=0.75):
    """Compute strictly feasible initial radii based on local geometry."""
    # Distance to walls
    wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Distance to other centers
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    r = np.minimum(wall, np.min(dists, axis=1) / 2.0)
    
    return r * shrink

def make_vector(centers, shrink=0.75):
    """Create optimization vector from centers."""
    r = get_feasible_radii(centers, shrink)
    return np.concatenate([centers[:, 0], centers[:, 1], r])

def generate_initial_configs():
    """Generate diverse, high-quality initial configurations."""
    inits = []
    
    # 1. Hexagonal lattices with various densities and rotations
    for r0 in [0.088, 0.092, 0.096, 0.100, 0.105]:
        for ang in np.linspace(-0.25, 0.25, 7):
            pts = []
            y = r0
            row = 0
            while len(pts) < N + 10:
                xs = r0 if row % 2 == 0 else 2*r0
                x = xs
                while x <= 1.0 - r0 and len(pts) < N + 10:
                    pts.append([x, y])
                    x += 2.0*r0
                y += np.sqrt(3.0)*r0
                row += 1
            pts = np.array(pts[:N])
            if ang != 0.0:
                pts -= 0.5
                c, s = np.cos(ang), np.sin(ang)
                pts = pts @ np.array([[c, -s], [s, c]]) + 0.5
            pts = np.clip(pts, 0.02, 0.98)
            inits.append(make_vector(pts, shrink=0.7))
            
    # 2. Force-relaxed dense scatters
    for seed in range(12):
        np.random.seed(seed + 50)
        pts = np.random.uniform(0.12, 0.88, (N, 2))
        # Repulsion relaxation to spread points evenly
        for _ in range(250):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 0.22 and d > 1e-5:
                        f = (0.22 - d) / d * 0.6
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.04
            pts = np.clip(pts, 0.04, 0.96)
        inits.append(make_vector(pts, shrink=0.65))
        
    return inits

def run_packing():
    """Optimizes packing of 26 circles in a unit square to maximize sum of radii."""
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    inits = generate_initial_configs()
    
    # Phase 1: Multi-start optimization
    for v0 in inits:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        best_v = inits[0]
        
    # Phase 2: Iterative perturbation & refinement to escape local minima
    current_v = best_v.copy()
    for step in range(35):
        np.random.seed(step + 3000)
        v_p = current_v.copy()
        
        # Decaying noise perturbation
        noise_scale = 0.006 * (1.0 - step / 35.0)
        v_p[:2*N] += np.random.uniform(-noise_scale, noise_scale, 2*N)
        v_p[:2*N] = np.clip(v_p[:2*N], 0.015, 0.985)
        
        # Index swap to break symmetry traps
        if np.random.rand() < 0.3:
            i, j = np.random.choice(N, 2, replace=False)
            v_p[:N][i], v_p[:N][j] = v_p[:N][j], v_p[:N][i]
            v_p[N:2*N][i], v_p[N:2*N][j] = v_p[N:2*N][j], v_p[N:2*N][i]
            
        # Shrink radii significantly to allow rearrangement, then recompute feasible radii
        centers_p = np.column_stack([v_p[:N], v_p[N:2*N]])
        v_p[2*N:] = get_feasible_radii(centers_p, shrink=0.8)
        
        try:
            res = minimize(objective, v_p, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v.copy()
        except Exception:
            continue
            
    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Phase 3: Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0 - cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0 - cy))
    cr = np.maximum(cr, 0.0)
    
    # 2. Enforce non-overlap constraints iteratively with minimal shrinkage
    for _ in range(25):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if cr[i] + cr[j] > d - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed:
            break
            
    centers = np.column_stack([cx, cy])
    return centers, cr, float(np.sum(cr))
