# sol_000128 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000091 (state 364131c7) state=9970da69 sum of radii=2.630179 correctness=1.0
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
    """Inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: circles inside [0,1]x[0,1]
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints (squared distance for smooth gradients)
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c_pair = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return np.concatenate([c, c_pair])

def compute_feasible_radii(centers):
    """Compute strictly feasible radii for given centers."""
    r = np.full(N, 0.5)
    for i in range(N):
        mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        for j in range(N):
            if i != j:
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d/2.0 < mr:
                    mr = d/2.0
        r[i] = mr
    return r * 0.85

def generate_starts():
    """Generate diverse initial configurations."""
    starts = []
    
    # 1. Hexagonal lattices with variations
    for r0 in [0.088, 0.092, 0.096, 0.100]:
        for shift_y in [-0.02, 0.0, 0.02]:
            for ang in [0.0, 0.12, -0.12]:
                pts = []
                y = r0 + shift_y
                row = 0
                while len(pts) < N + 5:
                    x = r0 + (row % 2) * r0
                    while x <= 1.0 - r0 and len(pts) < N + 5:
                        pts.append([x, y])
                        x += 2.0 * r0
                    y += r0 * np.sqrt(3.0)
                    row += 1
                pts = np.array(pts[:N])
                if ang != 0.0:
                    pts -= 0.5
                    c, s = np.cos(ang), np.sin(ang)
                    pts = pts @ np.array([[c, -s], [s, c]])
                    pts += 0.5
                pts = np.clip(pts, 0.02, 0.98)
                starts.append(np.concatenate([pts[:, 0], pts[:, 1], compute_feasible_radii(pts)]))

    # 2. Repelled random configurations
    for seed in range(20):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        for _ in range(400):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 0.25 and d > 1e-5:
                        f = (0.25 - d) / d * 0.5
                        diff = pts[i] - pts[j]
                        forces[i] += f * diff
                        forces[j] -= f * diff
            pts += forces * 0.02
            pts = np.clip(pts, 0.05, 0.95)
        starts.append(np.concatenate([pts[:, 0], pts[:, 1], compute_feasible_radii(pts)]))
        
    return starts

def run_packing():
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_v = None
    best_sum = -1.0
    
    # Phase 1: Multi-start optimization
    starts = generate_starts()
    for v0 in starts:
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            s = -res.fun
            if s > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = starts[0]
        
    # Phase 2: Iterative refinement to escape local minima
    current_v = best_v
    for step in range(30):
        pert = current_v.copy()
        # Perturb centers
        noise = np.random.uniform(-0.005, 0.005, 2 * N)
        pert[:2 * N] += noise
        pert[:2 * N] = np.clip(pert[:2 * N], 0.01, 0.99)
        
        # Recompute feasible radii for perturbed centers to guarantee valid restart
        centers_pert = np.column_stack([pert[:N], pert[N:2*N]])
        pert[2 * N:] = compute_feasible_radii(centers_pert) * 0.92
        
        try:
            res = minimize(objective, pert, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            s = -res.fun
            if s > best_sum:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
                    current_v = best_v
        except Exception:
            pass
            
    # Extract final configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Strict post-processing to guarantee validator compliance
    for _ in range(5):
        for i in range(N):
            cr[i] = min(cr[i], cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if cr[i] + cr[j] > d - 1e-9:
                    shrink = (cr[i] + cr[j] - d) / 2.0 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    
    centers = np.column_stack([cx, cy])
    return centers, cr, float(np.sum(cr))
