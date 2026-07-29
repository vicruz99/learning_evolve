# sol_000129 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000091 (state 364131c7) state=be205fd5 sum of radii=2.630091 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v, i_idx, j_idx):
    """Inequality constraints: boundaries and non-overlap (squared distances)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints (squared for smoother gradients)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    c_pair = dx**2 + dy**2 - (r[i_idx] + r[j_idx])**2
    
    return np.concatenate([c, c_pair])

def compute_feasible_radii(centers):
    """Compute strictly feasible initial radii based on local geometry."""
    r = np.full(N, 0.5)
    # 1. Boundary limits
    for i in range(N):
        r[i] = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        
    # 2. Pairwise limits
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            half_d = d * 0.5
            if half_d < r[i]: r[i] = half_d
            if half_d < r[j]: r[j] = half_d
            
    return r

def relax_config(pts, steps=600):
    """Spread points using repulsive forces to create a good initial packing."""
    pts = pts.copy()
    for _ in range(steps):
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = pts[i,0] - pts[j,0]
                dy = pts[i,1] - pts[j,1]
                d2 = dx*dx + dy*dy
                if d2 < 0.06 and d2 > 1e-6:
                    d = np.sqrt(d2)
                    f = (0.25 - d) / d
                    forces[i] += f * np.array([dx, dy])
                    forces[j] -= f * np.array([dx, dy])
                    
        # Wall repulsion
        forces[:, 0] += np.where(pts[:, 0] < 0.12, (0.12 - pts[:, 0])*5.0, 0.0)
        forces[:, 0] -= np.where(pts[:, 0] > 0.88, (pts[:, 0] - 0.88)*5.0, 0.0)
        forces[:, 1] += np.where(pts[:, 1] < 0.12, (0.12 - pts[:, 1])*5.0, 0.0)
        forces[:, 1] -= np.where(pts[:, 1] > 0.88, (pts[:, 1] - 0.88)*5.0, 0.0)
        
        pts += forces * 0.04
        pts = np.clip(pts, 0.03, 0.97)
    return pts

def generate_starts():
    """Generate diverse initial configurations."""
    starts = []
    np.random.seed(42)
    
    # 1. Relaxed random placements
    for _ in range(8):
        pts = np.random.uniform(0.25, 0.75, (N, 2))
        starts.append(relax_config(pts))
        
    # 2. Hexagonal grids with varying densities
    for r0 in [0.085, 0.095, 0.105, 0.115]:
        pts = []
        y = r0
        row = 0
        while len(pts) < N + 5:
            x_start = r0 + (row % 2) * r0
            x = x_start
            while x <= 1 - r0 and len(pts) < N + 5:
                pts.append([x, y])
                x += 2 * r0
            y += r0 * np.sqrt(3)
            row += 1
        if len(pts) >= N:
            starts.append(np.array(pts[:N]))
            
    # 3. Square grids
    for s in [0.14, 0.15, 0.16, 0.17]:
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([s + i*2*s, s + j*2*s])
        if len(pts) >= N:
            starts.append(np.array(pts[:N]))
            
    # 4. Corner-focused + relaxed center
    for seed in range(5):
        np.random.seed(seed + 1000)
        pts = np.array([[0.12,0.12], [0.88,0.12], [0.12,0.88], [0.88,0.88]])
        rest = np.random.uniform(0.25, 0.75, (N-4, 2))
        pts = np.vstack([pts, rest])
        starts.append(relax_config(pts))
        
    return starts

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    i_idx, j_idx = np.triu_indices(N, k=1)
    bounds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    cons = {'type': 'ineq', 'fun': constraints, 'args': (i_idx, j_idx)}
    
    best_v = None
    best_sum = -1.0
    
    starts = generate_starts()
    
    # Phase 1: Multi-start optimization
    for pts in starts:
        r_init = compute_feasible_radii(pts) * 0.88
        v0 = np.concatenate([pts[:,0], pts[:,1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            s = -res.fun
            if s > best_sum:
                if np.min(constraints(res.x, i_idx, j_idx)) >= -1e-6:
                    best_sum = s
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Perturbation refinement to escape local minima
    if best_v is not None:
        for step in range(25):
            np.random.seed(step + 2000)
            v_pert = best_v.copy()
            v_pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.02, 0.98)
            v_pert[2*N:] *= 0.94
            
            # Guarantee strict feasibility before restart
            c_pert = v_pert[:2*N].reshape(N, 2)
            v_pert[2*N:] = compute_feasible_radii(c_pert) * 0.85
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                s = -res.fun
                if s > best_sum:
                    if np.min(constraints(res.x, i_idx, j_idx)) >= -1e-6:
                        best_sum = s
                        best_v = res.x.copy()
            except Exception:
                pass
                
    # Extract optimal configuration
    cx = best_v[:N]
    cy = best_v[N:2*N]
    cr = best_v[2*N:].copy()
    
    # Strict post-processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    cr = np.minimum(cr, np.minimum(cx, 1.0-cx))
    cr = np.minimum(cr, np.minimum(cy, 1.0-cy))
    cr = np.maximum(cr, 0.0)
    
    # 2. Enforce non-overlap strictly with iterative shrinkage
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                if cr[i] + cr[j] > d - 1e-9:
                    shrink = (cr[i] + cr[j] - d)/2.0 + 1e-9
                    cr[i] = max(0.0, cr[i] - shrink)
                    cr[j] = max(0.0, cr[j] - shrink)
                    changed = True
        if not changed: break
        
    centers = np.column_stack((cx, cy))
    return centers, cr, float(np.sum(cr))
