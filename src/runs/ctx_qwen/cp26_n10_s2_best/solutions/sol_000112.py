# sol_000112 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000055 (state f6ce444f) state=83f25ed6 sum of radii=2.629619 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pair indices for upper triangular pairwise constraints
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def objective(v):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: circles inside [0,1]^2
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: dist >= r_i + r_j
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[PAIR_I] + r[PAIR_J]
    c_pair = dist - r_sum
    
    return np.concatenate([c_bound, c_pair])

def generate_hex_lattice(row_counts, r_base, angle_deg=0.0):
    """Generate a hexagonal lattice configuration with specified row counts."""
    pts = []
    y = r_base
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    row_idx = 0
    for count in row_counts:
        x_start = r_base if row_idx % 2 == 0 else 2.0 * r_base
        for k in range(count):
            x = x_start + k * 2.0 * r_base
            pts.append([x, y])
        y += np.sqrt(3) * r_base
        row_idx += 1
        
    pts = np.array(pts[:N])
    
    if angle_deg != 0.0:
        # Rotate around center (0.5, 0.5)
        pts = pts - 0.5
        rot_x = pts[:, 0] * cos_a - pts[:, 1] * sin_a
        rot_y = pts[:, 0] * sin_a + pts[:, 1] * cos_a
        pts[:, 0] = rot_x + 0.5
        pts[:, 1] = rot_y + 0.5
        
    return pts

def get_feasible_r(centers):
    """Compute strictly feasible initial radii for given centers."""
    r = np.full(N, 0.5)
    for i in range(N):
        # Distance to boundaries
        r[i] = min(centers[i,0], 1.0 - centers[i,0], 
                   centers[i,1], 1.0 - centers[i,1])
        # Distance to other centers
        for j in range(N):
            if i == j: continue
            d = np.hypot(centers[i,0] - centers[j,0], 
                         centers[i,1] - centers[j,1])
            r[i] = min(r[i], d / 2.0)
    return r * 0.85  # 85% leaves room for optimizer to expand

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    configs_to_try = []
    
    # 1. Diverse Hexagonal Lattices
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], 
        [4,6,6,6,4], [7,5,5,5,4], [6,6,5,5,4]
    ]
    angles = [0, 1, -1, 2, -2, 3, -3]
    
    for r0 in [0.095, 0.100, 0.105]:
        for pat in patterns:
            for ang in angles:
                try:
                    pts = generate_hex_lattice(pat, r0, ang)
                    # Filter points that fall outside safe margin
                    mask = (pts[:,0] >= 0.02) & (pts[:,0] <= 0.98) & \
                           (pts[:,1] >= 0.02) & (pts[:,1] <= 0.98)
                    if np.sum(mask) < N: continue
                    pts = pts[mask][:N]
                    r_init = get_feasible_r(pts)
                    configs_to_try.append(np.concatenate([pts[:,0], pts[:,1], r_init]))
                except Exception:
                    pass
                    
    # 2. Square Grid Lattices
    for r0 in [0.095, 0.100, 0.105]:
        pts = []
        y = r0
        while len(pts) < N + 5:
            x = r0
            while x <= 1 - r0 and len(pts) < N + 5:
                pts.append([x, y])
                x += 2 * r0
            y += 2 * r0
        pts = np.array(pts[:N])
        r_init = get_feasible_r(pts)
        configs_to_try.append(np.concatenate([pts[:,0], pts[:,1], r_init]))
        
    # 3. Random Dense Placements
    for seed in range(15):
        np.random.seed(seed)
        pts = []
        while len(pts) < N:
            p = np.random.uniform(0.05, 0.95, 2)
            if all(np.hypot(p[0]-q[0], p[1]-q[1]) > 0.14 for q in pts):
                pts.append(p)
        pts = np.array(pts[:N])
        r_init = get_feasible_r(pts)
        configs_to_try.append(np.concatenate([pts[:,0], pts[:,1], r_init]))

    # Primary Multi-Start Optimization
    for x0 in configs_to_try:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, 
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_sum:
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-8:
                    best_sum = -res.fun
                    best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        best_v = configs_to_try[0]
        
    # Adaptive Refinement: Escape local minima by shrinking and perturbing
    current_v = best_v
    for step in range(10):
        scale = 0.996 - step * 0.001
        perturbed = current_v.copy()
        perturbed[2*N:] *= scale
        perturbed[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
        perturbed[:2*N] = np.clip(perturbed[:2*N], 0.02, 0.98)
        
        try:
            res = minimize(objective, perturbed, method='SLSQP', bounds=bounds,
                           constraints=cons, 
                           options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if -res.fun > best_sum:
                if np.min(constraints(res.x)) >= -1e-8:
                    best_sum = -res.fun
                    best_v = res.x.copy()
                    current_v = best_v
        except Exception:
            pass
            
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:]
    
    # Strict Post-Processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    for i in range(N):
        radii[i] = min(radii[i], centers[i,0], 1.0 - centers[i,0], 
                       centers[i,1], 1.0 - centers[i,1])
        radii[i] = max(radii[i], 0.0)
        
    # 2. Enforce non-overlap strictly with iterative shrinkage
    for _ in range(30):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0] - centers[j,0], 
                             centers[i,1] - centers[j,1])
                if radii[i] + radii[j] > d:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-9
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
