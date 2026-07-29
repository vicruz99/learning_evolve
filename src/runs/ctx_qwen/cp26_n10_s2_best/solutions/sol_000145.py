# sol_000145 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000112 (state 83f25ed6) state=fc2780c2 sum of radii=2.602283 correctness=1.0
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
    """Compute inequality constraints: boundaries and non-overlap (squared)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: circles must be inside [0, 1]x[0, 1]
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints (squared for numerical stability)
    dx = x[PAIR_I] - x[PAIR_J]
    dy = y[PAIR_I] - y[PAIR_J]
    c_pair = dx**2 + dy**2 - (r[PAIR_I] + r[PAIR_J])**2
    
    return np.concatenate([c_bound, c_pair])

def generate_lattice_config(row_counts, r_base, angle_deg, shift_x, shift_y):
    """Generate a hexagonal lattice configuration with specified row counts."""
    pts = []
    y = r_base + shift_y
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    row_idx = 0
    for count in row_counts:
        x_start = r_base + shift_x + (row_idx % 2) * r_base
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

def compute_initial_radii(centers):
    """Compute strictly feasible initial radii for given centers."""
    r = np.full(N, 0.5)
    for i in range(N):
        # Distance to boundaries
        r[i] = min(centers[i,0], 1.0 - centers[i,0], 
                   centers[i,1], 1.0 - centers[i,1])
        # Distance to other centers
        for j in range(i + 1, N):
            d = np.hypot(centers[i,0] - centers[j,0], 
                         centers[i,1] - centers[j,1])
            val = d / 2.0
            if val < r[i]:
                r[i] = val
            if val < r[j]:
                r[j] = val
                
    # Scale down to guarantee strict feasibility initially
    r = r * 0.85
    return np.maximum(r, 1e-5)

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.25)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    configs = []
    # Patterns that sum to 26 circles, known to be efficient for square packing
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,6,6,4], 
        [7,5,5,5,4], [6,6,5,5,4], [5,5,6,5,5]
    ]
    
    # Systematically explore parameter space for hexagonal lattices
    for pat in patterns:
        for r0 in [0.095, 0.100, 0.105]:
            for ang in [0, 3, -3, 5, -5, 8, -8]:
                for sx in [-0.02, 0.0, 0.02]:
                    for sy in [-0.02, 0.0, 0.02]:
                        try:
                            pts = generate_lattice_config(pat, r0, ang, sx, sy)
                            # Filter points that fall outside safe margin
                            mask = (pts[:,0] >= 0.02) & (pts[:,0] <= 0.98) & \
                                   (pts[:,1] >= 0.02) & (pts[:,1] <= 0.98)
                            if np.sum(mask) < N:
                                continue
                            pts = pts[mask][:N]
                            r_init = compute_initial_radii(pts)
                            configs.append(np.concatenate([pts[:,0], pts[:,1], r_init]))
                        except Exception:
                            pass
                            
    # Add diverse random starts to cover non-lattice optima
    for seed in range(15):
        np.random.seed(seed)
        pts = np.random.uniform(0.15, 0.85, (N, 2))
        r_init = compute_initial_radii(pts)
        configs.append(np.concatenate([pts[:,0], pts[:,1], r_init]))
        
    # Sample a representative subset to keep runtime reasonable
    np.random.seed(42)
    np.random.shuffle(configs)
    configs = configs[:60]
    
    # Phase 1: Multi-start optimization
    for x0 in configs:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            s = -res.fun
            c_val = constraints(res.x)
            # Accept if sufficiently feasible and better than current best
            if np.min(c_val) >= -1e-6 and s > best_sum:
                best_sum = s
                best_v = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Adaptive Refinement to escape local minima
    if best_v is not None:
        for step in range(15):
            np.random.seed(step + 5000)
            v_pert = best_v.copy()
            # Slightly shrink radii to create breathing room for center movement
            v_pert[2*N:] *= 0.992
            # Perturb centers
            v_pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            v_pert[:2*N] = np.clip(v_pert[:2*N], 0.02, 0.98)
            
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
                s = -res.fun
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-6 and s > best_sum:
                    best_sum = s
                    best_v = res.x.copy()
            except Exception:
                pass
                
    # Extract final configuration
    centers = np.column_stack((best_v[:N], best_v[N:2*N]))
    radii = best_v[2*N:].copy()
    
    # Phase 3: Strict Post-Processing to guarantee validator compliance
    # 1. Enforce boundary constraints strictly
    radii = np.minimum(radii, np.minimum(centers[:,0], 1.0 - centers[:,0]))
    radii = np.minimum(radii, np.minimum(centers[:,1], 1.0 - centers[:,1]))
    radii = np.maximum(radii, 0.0)
    
    # 2. Enforce non-overlap strictly with iterative shrinkage
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i,0] - centers[j,0], 
                             centers[i,1] - centers[j,1])
                if radii[i] + radii[j] > d - 1e-10:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-10
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return centers, radii, float(np.sum(radii))
