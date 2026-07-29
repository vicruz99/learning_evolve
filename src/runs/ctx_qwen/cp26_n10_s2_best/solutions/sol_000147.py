# sol_000147 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000112 (state 83f25ed6) state=4948e2da sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap (squared for stability)."""
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
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist_sq = dx**2 + dy**2
    r_sum_sq = (r[PAIR_I] + r[PAIR_J])**2
    
    return np.concatenate([c, dist_sq - r_sum_sq])

def get_initial_r(centers):
    """Compute strictly feasible initial radii for given centers."""
    r = np.zeros(N)
    for i in range(N):
        mr = min(centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
        for j in range(N):
            if i == j: continue
            d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
            mr = min(mr, d / 2.0)
        r[i] = mr * 0.65  # 65% leaves significant room for optimizer expansion
    return r

def repulsion_layout(n, seed=0, iters=500):
    """Generate a well-separated configuration using force-directed relaxation."""
    np.random.seed(seed)
    pts = np.random.uniform(0.2, 0.8, (n, 2))
    alpha = 0.04
    target_dist = 0.16
    
    for _ in range(iters):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i,0] - pts[j,0]
                dy = pts[i,1] - pts[j,1]
                d2 = dx*dx + dy*dy
                if d2 < target_dist**2 and d2 > 1e-6:
                    d = np.sqrt(d2)
                    f = (target_dist - d) / d
                    forces[i,0] += f * dx
                    forces[i,1] += f * dy
                    forces[j,0] -= f * dx
                    forces[j,1] -= f * dy
            
            # Boundary repulsion
            for dim in range(2):
                if pts[i, dim] < 0.05:
                    forces[i, dim] += 10.0 * (0.05 - pts[i, dim])
                if pts[i, dim] > 0.95:
                    forces[i, dim] -= 10.0 * (pts[i, dim] - 0.95)
                    
        pts += forces * alpha
        pts = np.clip(pts, 0.01, 0.99)
        alpha *= 0.995
    return pts

def generate_hex(r0, ang_deg):
    """Generate a hexagonal lattice with optional rotation."""
    pts = []
    y = r0
    row = 0
    while len(pts) < N + 5:
        x_start = r0 if row % 2 == 0 else 2.0 * r0
        x = x_start
        while x <= 1.0 - r0 and len(pts) < N + 5:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
        
    pts = np.array(pts[:N])
    if ang_deg != 0:
        pts = pts - 0.5
        a = np.deg2rad(ang_deg)
        ca, sa = np.cos(a), np.sin(a)
        pts = np.column_stack([pts[:,0]*ca - pts[:,1]*sa, pts[:,0]*sa + pts[:,1]*ca]) + 0.5
    return pts

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    configs = []
    
    # 1. Force-directed repulsion layouts
    for s in range(12):
        configs.append(repulsion_layout(N, seed=s))
        
    # 2. Rotated hexagonal lattices
    for r0 in [0.095, 0.100, 0.105]:
        for ang in [0, 1, -1, 2, -2]:
            configs.append(generate_hex(r0, ang))
            
    # Phase 1: Multi-start optimization
    for centers in configs:
        r_init = get_initial_r(centers)
        v0 = np.concatenate([centers[:,0], centers[:,1], r_init])
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_sum:
                if np.min(constraints(res.x)) >= -1e-7:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Perturbation refinement to escape local minima
    if best_v is not None:
        curr = best_v
        for step in range(20):
            np.random.seed(step + 1000)
            v_p = curr.copy()
            v_p[2*N:] *= 0.97  # Shrink radii to create breathing room
            v_p[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            v_p[:2*N] = np.clip(v_p[:2*N], 0.01, 0.99)
            
            try:
                res = minimize(objective, v_p, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                if -res.fun > best_sum:
                    if np.min(constraints(res.x)) >= -1e-7:
                        best_sum = -res.fun
                        best_v = res.x.copy()
                        curr = best_v
            except Exception:
                pass
                
    # Extract optimal configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:]
    
    # Phase 3: Strict Post-Processing for Validator Compliance
    # 1. Enforce boundary constraints
    for i in range(N):
        radii[i] = min(radii[i], centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
        radii[i] = max(radii[i], 0.0)
        
    # 2. Enforce non-overlap strictly with iterative shrinkage
    for _ in range(25):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                if radii[i] + radii[j] > d - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
